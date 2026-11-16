/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR3 decode on Vulkan compute, checked against the CPU reference (prototype, docs/WNR3-VULKAN.md).
 *
 * The host validates each record (wex_nrx_parse) and its model (wex_nr2_model_open), uploads the record
 * bytes, the WNM2 blob and a scale symbol table built from it, a 32-word plan and the stream offsets, and
 * runs the two passes of wnr3.comp
 * (scales.spv, then weights.spv) for every record of a batch in one command buffer. The decoded bytes land
 * in a storage buffer the host never maps (device-local memory where the device has it); they are copied to
 * a host-visible buffer only to be compared here with wex_nr2_decode (c/wex_neural_rans.h).
 *
 *   wnr3_vk selftest [CASE]          synthetic records from the reference encoder: every flag, chunk sizes
 *                                    at both ends, scale counts that are not multiples of 4, and seven
 *                                    records that must be refused, each with its error bits. CASE (0 to 11)
 *                                    takes one of the twelve records alone
 *   wnr3_vk decode FILE.wex4 ...     every WNR3 record of each file must decode, with the reference's bytes
 *   wnr3_vk reject FILE.wex4 ...     each file must hold a WNR3 record that the reference refuses; the
 *                                    Vulkan verdict must be the same for every record
 *   wnr3_vk mutate N SEED FILE.wex4 ...   N copies of each WNR3 record with one byte changed: same verdict
 *                                    as the reference, and the same bytes when both accept
 *   wnr3_vk bench RUNS FILE.wex4 [RECORDS]   the first RECORDS WNR3 records of the file (default: as many as
 *                                    one batch holds) decoded RUNS times on the device and on the CPU, with
 *                                    the time of each part; the bytes must be the reference's (bench.sh)
 *
 * A file is not loaded whole: the header page, the index and the model blobs are read once, and each record
 * when its batch is filled, with 64-bit offsets. A batch is cut at MAX_BATCH records or when the next record
 * would take the stored or the decoded bytes of the batch above the batch limit.
 *
 * Environment: WNR3_VK_SHADERS = folder of scales.spv and weights.spv (default: the working directory);
 * WNR3_VK_DEVICE = index of the physical device (default 0), or a part of its name, ASCII letters compared
 * without case (NVIDIA, llvmpipe); WNR3_VK_STAGE=1 uploads the records, the model blobs, the plans and the offsets
 * through host-visible staging buffers into device-local ones, WNR3_VK_STAGE=0 lets the shader read the
 * host-visible buffers the host wrote (default: 1 when the device has memory the host cannot map, else 0);
 * WNR3_VK_ROBUST=1 enables the device feature robustBufferAccess (default: no feature); WNR3_VK_VALIDATE=1
 * turns the Khronos validation layer on and makes any message of it an error; WNR3_VK_BATCH_BYTES = batch
 * limit in bytes (default 2^28; the device's maxStorageBufferRange when that is smaller); WNR3_VK_EVERY = N
 * takes one WNR3 record in N of each file (default 1: all); WNR3_VK_BENCH_CPU=0 leaves the CPU decoders out
 * of the times of `bench`; WNR3_VK_SHADERS_B = the folder of a second form of the two shaders, which `bench`
 * then times against the first in pairs on the same batch (and checks against the reference too).
 * Exit 0 on success, 1 on a difference, 2 on a setup error. */
#ifndef _WIN32
#define _FILE_OFFSET_BITS 64       /* a 64-bit off_t for fseeko and ftello */
#define _POSIX_C_SOURCE 200112L    /* which -std=c99 would hide */
#endif
#include <stdio.h>
#include <vulkan/vulkan.h>
#include "wex4.h"
#ifdef WNR3_VK_FAST
#include "wex_fast_decode.h"   /* wex4/research/fast-cpu-decode: the bench mode times it too */
#endif
#ifdef _WIN32
#include <windows.h>
#define seek_file(f, at) _fseeki64(f, (long long)(at), SEEK_SET)
#define seek_end(f) _fseeki64(f, 0, SEEK_END)
#define tell_file(f) ((int64_t)_ftelli64(f))
#else
#include <sys/types.h>
#include <time.h>
#define seek_file(f, at) fseeko(f, (off_t)(at), SEEK_SET)
#define seek_end(f) fseeko(f, 0, SEEK_END)
#define tell_file(f) ((int64_t)ftello(f))
#endif

#define PLAN_WORDS 32u
#define GROUP 32u            /* local_size_x of wnr3.comp */
#define SCALE_TABLE_BYTES (3u * WEX_NR2_TOTAL)                    /* the scale symbol of each slot, per matrix */
#define MODEL_SLOT_BYTES (WEX_NR2_MODEL_BYTES + SCALE_TABLE_BYTES)  /* a model in the tab buffer */
#define MAX_BATCH 64u
#define BATCH_BYTES (UINT64_C(1) << 28)
#define ERR_STRUCT 1u
#define ERR_HASH 2u
#define ERR_CEIL 4u
#define ERR_CZ 8u

#define DIE(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
#define VK(call) do { VkResult vk_r = (call); if (vk_r != VK_SUCCESS) DIE("%s: VkResult %d", #call, (int)vk_r); } while (0)

typedef struct { VkBuffer buffer; VkDeviceMemory memory; VkDeviceSize bytes; void *map; VkMemoryPropertyFlags flags; } Buffer;

typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
    VkPhysicalDevice physical;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceLimits limits;
    VkDevice device;
    VkQueue queue;
    uint32_t family;
    VkCommandPool pool;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorPool descriptors;
    VkShaderModule module[4];
    VkPipeline pipeline[4];   /* 2 * form + 0 scales, 2 * form + 1 weights */
    uint32_t forms;           /* 2 when WNR3_VK_SHADERS_B names a second form of the shader (bench), else 1 */
    uint32_t form;            /* the form slab_run binds */
    int validate;
    int staged;               /* the inputs go through staging buffers into device-local ones (WNR3_VK_STAGE) */
    double pipeline_ms;       /* both compute pipelines, shader modules included */
    uint32_t timestamp_bits;  /* valid bits of the queue's timestamps (0: the queue has none) */
} Gpu;

static unsigned long validation_messages;

/* One record to decode: stored bytes that passed wex_nrx_parse, its layout and its validated model blob. */
typedef struct {
    const unsigned char *stored;
    uint32_t stored_bytes;
    WexNr2Record view;
    WexNr2Layout layout;
    const unsigned char *model;       /* WEX_NR2_MODEL_BYTES */
    /* results */
    uint32_t error;                   /* the record's error word */
    uint32_t scale_at[3], weight_at[3];   /* byte offsets of the decoded regions in the read-back copy */
} Job;

static void *xmalloc(size_t n) {
    void *p = calloc(n ? n : 1u, 1u);
    if (!p) DIE("out of memory (%lu bytes)", (unsigned long)n);
    return p;
}

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return (double)ticks.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) DIE("clock_gettime");
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
#endif
}

/* a whole small file (a SPIR-V module); containers go through container_open */
static unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *data;
    long n;
    if (!f) DIE("%s: cannot open", path);
    if (fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) DIE("%s: cannot seek", path);
    data = (unsigned char *)xmalloc((size_t)n + 4u);
    if (fread(data, 1u, (size_t)n, f) != (size_t)n) DIE("%s: short read", path);
    fclose(f);
    *size = (size_t)n;
    return data;
}

/* ---- Vulkan set-up ------------------------------------------------------------------------------------ */

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)severity; (void)types; (void)user;
    validation_messages++;
    fprintf(stderr, "validation: %s\n", data->pMessage);
    return VK_FALSE;
}

static uint32_t memory_type(const Gpu *g, uint32_t allowed, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    uint32_t i;
    for (i = 0; i < g->memory.memoryTypeCount; i++)
        if ((allowed >> i & 1u) && (g->memory.memoryTypes[i].propertyFlags & want) == want &&
            !(g->memory.memoryTypes[i].propertyFlags & avoid)) return i;
    return UINT32_MAX;
}

/* host = 1: host-visible and coherent, mapped. host = 2: the same, cached where the device has such a type
 * (memory the host reads). host = 0: memory the host does not map, device-local and
 * not host-visible where such a type exists (a software device has none: any type then). */
static void buffer_create(const Gpu *g, VkDeviceSize bytes, VkBufferUsageFlags usage, int host, Buffer *b) {
    VkBufferCreateInfo info;
    VkMemoryRequirements need;
    VkMemoryAllocateInfo alloc;
    uint32_t type;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = b->bytes = (bytes + 3u) & ~(VkDeviceSize)3u;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(g->device, &info, NULL, &b->buffer));
    vkGetBufferMemoryRequirements(g->device, b->buffer, &need);
    if (host) {
        const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        type = host == 2 ? memory_type(g, need.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0) : UINT32_MAX;
        if (type == UINT32_MAX) type = memory_type(g, need.memoryTypeBits, visible, 0);
    } else {
        type = memory_type(g, need.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (type == UINT32_MAX) type = memory_type(g, need.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (type == UINT32_MAX) type = memory_type(g, need.memoryTypeBits, 0, 0);
    }
    if (type == UINT32_MAX) DIE("no memory type for a buffer of %lu bytes", (unsigned long)bytes);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = type;
    b->flags = g->memory.memoryTypes[type].propertyFlags;
    VK(vkAllocateMemory(g->device, &alloc, NULL, &b->memory));
    VK(vkBindBufferMemory(g->device, b->buffer, b->memory, 0));
    b->map = NULL;
    if (host) VK(vkMapMemory(g->device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map));
}

static void buffer_destroy(const Gpu *g, Buffer *b) {
    vkDestroyBuffer(g->device, b->buffer, NULL);
    vkFreeMemory(g->device, b->memory, NULL);   /* unmaps */
    memset(b, 0, sizeof(*b));
}

static VkShaderModule load_shader(const Gpu *g, const char *folder, const char *name) {
    char path[1024];
    size_t size;
    unsigned char *code;
    VkShaderModuleCreateInfo info;
    VkShaderModule module;
    if (!folder || !folder[0]) folder = ".";
    if (snprintf(path, sizeof(path), "%s/%s", folder, name) >= (int)sizeof(path)) DIE("WNR3_VK_SHADERS is too long");
    code = read_file(path, &size);
    if (!size || (size & 3u)) DIE("%s: not SPIR-V", path);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = (const uint32_t *)(const void *)code;   /* calloc memory: aligned for uint32_t */
    VK(vkCreateShaderModule(g->device, &info, NULL, &module));
    free(code);
    return module;
}

/* c with an ASCII capital letter turned into its small letter; any other byte as it is */
static int ascii_lower(char c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

/* 1 when name holds part, ASCII letters compared without case, every other byte as it is */
static int name_has(const char *name, const char *part) {
    size_t i, k;
    for (i = 0; name[i]; i++) {
        for (k = 0; part[k] && name[i + k] && ascii_lower(name[i + k]) == ascii_lower(part[k]); k++) {}
        if (!part[k]) return 1;
    }
    return 0;
}

static void gpu_open(Gpu *g) {
    static const char *const layer = "VK_LAYER_KHRONOS_validation";
    static const char *const debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    const char *choice = getenv("WNR3_VK_DEVICE"), *validate = getenv("WNR3_VK_VALIDATE"), *stage = getenv("WNR3_VK_STAGE"),
               *robust = getenv("WNR3_VK_ROBUST");
    double start;
    VkApplicationInfo app;
    VkInstanceCreateInfo instance;
    VkPhysicalDevice devices[16];
    VkPhysicalDeviceProperties properties;
    VkQueueFamilyProperties families[32];
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo device;
    VkPhysicalDeviceFeatures features, wanted;
    VkCommandPoolCreateInfo pool;
    VkDescriptorSetLayoutBinding bindings[6];
    VkDescriptorSetLayoutCreateInfo set_layout;
    VkPushConstantRange push;
    VkPipelineLayoutCreateInfo layout;
    VkDescriptorPoolSize pool_size;
    VkDescriptorPoolCreateInfo descriptors;
    uint32_t count = 16u, families_count = 32u, i, index = 0;
    float priority = 1.0f;
    static const char *const names[2] = {"scales.spv", "weights.spv"};
    const char *folders[2];

    memset(g, 0, sizeof(*g));
    g->validate = validate && validate[0] == '1';

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "wnr3_vk";
    app.apiVersion = VK_API_VERSION_1_0;
    memset(&instance, 0, sizeof(instance));
    instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance.pApplicationInfo = &app;
    if (g->validate) {
        instance.enabledLayerCount = 1u; instance.ppEnabledLayerNames = &layer;
        instance.enabledExtensionCount = 1u; instance.ppEnabledExtensionNames = &debug_extension;
    }
    VK(vkCreateInstance(&instance, NULL, &g->instance));
    if (g->validate) {
        VkDebugUtilsMessengerCreateInfoEXT info;
        PFN_vkCreateDebugUtilsMessengerEXT create = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g->instance, "vkCreateDebugUtilsMessengerEXT");
        if (!create) DIE("the validation layer is on but VK_EXT_debug_utils is missing");
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = on_message;
        VK(create(g->instance, &info, NULL, &g->messenger));
    }
    {
        VkResult r = vkEnumeratePhysicalDevices(g->instance, &count, devices);
        if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !count) DIE("no Vulkan device");
    }
    if (choice && choice[0] && !choice[strspn(choice, "0123456789")]) {
        index = (uint32_t)strtoul(choice, NULL, 10);
        if (index >= count) DIE("WNR3_VK_DEVICE=%u but there are %u devices", (unsigned)index, (unsigned)count);
    } else if (choice && choice[0]) {   /* a part of the device name */
        for (index = 0; index < count; index++) {
            vkGetPhysicalDeviceProperties(devices[index], &properties);
            if (name_has(properties.deviceName, choice)) break;
        }
        if (index == count) fprintf(stderr, "WNR3_VK_DEVICE=%s: no device has that in its name\n", choice);
    }
    for (i = 0; i < count; i++) {
        char driver[48];
        uint32_t v;
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        /* the driver version is the vendor's own encoding; NVIDIA packs four parts in 10, 8, 8 and 6 bits */
        v = properties.driverVersion;
        if (properties.vendorID == 0x10deu)
            snprintf(driver, sizeof(driver), "%u.%u.%u.%u", (unsigned)(v >> 22), (unsigned)(v >> 14 & 255u),
                     (unsigned)(v >> 6 & 255u), (unsigned)(v & 63u));
        else
            snprintf(driver, sizeof(driver), "%u.%u.%u", (unsigned)VK_VERSION_MAJOR(v), (unsigned)VK_VERSION_MINOR(v),
                     (unsigned)VK_VERSION_PATCH(v));
        printf("device %u%s: %s (type %d, Vulkan %u.%u.%u, vendor 0x%04x, driver %s)\n", (unsigned)i, i == index ? " *" : "",
               properties.deviceName, (int)properties.deviceType, (unsigned)VK_VERSION_MAJOR(properties.apiVersion),
               (unsigned)VK_VERSION_MINOR(properties.apiVersion), (unsigned)VK_VERSION_PATCH(properties.apiVersion),
               (unsigned)properties.vendorID, driver);
    }
    if (index == count) exit(2);
    g->physical = devices[index];
    vkGetPhysicalDeviceProperties(g->physical, &properties);
    g->limits = properties.limits;
    vkGetPhysicalDeviceMemoryProperties(g->physical, &g->memory);
    /* staged upload: asked for, or the device has memory the host cannot map (a GPU with memory of its own) */
    g->staged = stage && stage[0] ? stage[0] == '1'
              : memory_type(g, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(g->physical, &families_count, families);
    for (i = 0; i < families_count && !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT); i++) {}
    if (i == families_count) DIE("the device has no compute queue");
    g->family = i;
    g->timestamp_bits = families[i].timestampValidBits;
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = g->family;
    queue.queueCount = 1u;
    queue.pQueuePriorities = &priority;
    memset(&device, 0, sizeof(device));
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.queueCreateInfoCount = 1u;
    device.pQueueCreateInfos = &queue;   /* no feature and no extension: core Vulkan 1.0 */
    if (robust && robust[0] == '1') {    /* asked for: out-of-bounds buffer accesses of a shader are made harmless */
        vkGetPhysicalDeviceFeatures(g->physical, &features);
        if (!features.robustBufferAccess) DIE("WNR3_VK_ROBUST=1 but the device has no robustBufferAccess");
        memset(&wanted, 0, sizeof(wanted));
        wanted.robustBufferAccess = VK_TRUE;
        device.pEnabledFeatures = &wanted;
        printf("robustBufferAccess is on\n");
    }
    VK(vkCreateDevice(g->physical, &device, NULL, &g->device));
    vkGetDeviceQueue(g->device, g->family, 0, &g->queue);

    memset(&pool, 0, sizeof(pool));
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = g->family;
    VK(vkCreateCommandPool(g->device, &pool, NULL, &g->pool));

    memset(bindings, 0, sizeof(bindings));
    for (i = 0; i < 6u; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1u;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    memset(&set_layout, 0, sizeof(set_layout));
    set_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout.bindingCount = 6u;
    set_layout.pBindings = bindings;
    VK(vkCreateDescriptorSetLayout(g->device, &set_layout, NULL, &g->set_layout));
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; push.offset = 0; push.size = 4u;
    memset(&layout, 0, sizeof(layout));
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1u; layout.pSetLayouts = &g->set_layout;
    layout.pushConstantRangeCount = 1u; layout.pPushConstantRanges = &push;
    VK(vkCreatePipelineLayout(g->device, &layout, NULL, &g->layout));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; pool_size.descriptorCount = 6u;
    memset(&descriptors, 0, sizeof(descriptors));
    descriptors.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptors.maxSets = 1u; descriptors.poolSizeCount = 1u; descriptors.pPoolSizes = &pool_size;
    VK(vkCreateDescriptorPool(g->device, &descriptors, NULL, &g->descriptors));

    folders[0] = getenv("WNR3_VK_SHADERS");
    folders[1] = getenv("WNR3_VK_SHADERS_B");
    if (!folders[0] || !folders[0][0]) folders[0] = ".";
    g->forms = folders[1] && folders[1][0] ? 2u : 1u;
    start = now_ms();
    for (i = 0; i < 2u * g->forms; i++) {
        VkComputePipelineCreateInfo info;
        g->module[i] = load_shader(g, folders[i / 2u], names[i % 2u]);
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = g->module[i];
        info.stage.pName = "main";
        info.layout = g->layout;
        VK(vkCreateComputePipelines(g->device, VK_NULL_HANDLE, 1u, &info, NULL, &g->pipeline[i]));
    }
    g->pipeline_ms = now_ms() - start;
}

static void gpu_close(Gpu *g) {
    uint32_t i;
    for (i = 0; i < 2u * g->forms; i++) {
        vkDestroyPipeline(g->device, g->pipeline[i], NULL);
        vkDestroyShaderModule(g->device, g->module[i], NULL);
    }
    vkDestroyDescriptorPool(g->device, g->descriptors, NULL);
    vkDestroyPipelineLayout(g->device, g->layout, NULL);
    vkDestroyDescriptorSetLayout(g->device, g->set_layout, NULL);
    vkDestroyCommandPool(g->device, g->pool, NULL);
    vkDestroyDevice(g->device, NULL);
    if (g->messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(g->instance, g->messenger, NULL);
    }
    vkDestroyInstance(g->instance, NULL);
}

/* ---- one batch ---------------------------------------------------------------------------------------- */

static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}
static uint64_t align16(uint64_t v) { return (v + 15u) & ~UINT64_C(15); }

/* bytes of the decoded buffer a record takes: six regions, each on a 16-byte boundary */
static uint64_t job_decoded_bytes(const Job *job) {
    uint64_t bytes = 0;
    uint32_t m;
    for (m = 0; m < 3u; m++) bytes += align16(job->layout.scale_count[m]) + align16(job->layout.symbol_count[m] / 2u);
    return bytes;
}

/* the most stored bytes, and the most decoded bytes, one batch may hold: one storage buffer each, with
 * 32-bit byte offsets in the shader */
static uint64_t batch_limit(const Gpu *g) {
    const char *text = getenv("WNR3_VK_BATCH_BYTES");
    uint64_t limit = text && text[0] ? (uint64_t)strtoull(text, NULL, 10) : BATCH_BYTES;
    if (limit > g->limits.maxStorageBufferRange) limit = g->limits.maxStorageBufferRange;
    if (limit > UINT32_MAX - 4095u) limit = UINT32_MAX - 4095u;
    return limit;
}

static void barrier(VkCommandBuffer cmd, VkBuffer buffer, VkAccessFlags from, VkAccessFlags to,
                    VkPipelineStageFlags from_stage, VkPipelineStageFlags to_stage) {
    VkBufferMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = from; b.dstAccessMask = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer; b.offset = 0; b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, from_stage, to_stage, 0, 0, NULL, 1u, &b, 0, NULL);
}

/* The buffers of one batch on the device, and what the shader needs to find them. */
typedef struct {
    Buffer rec, tab, plan, offs, out, err, back;
    Buffer stage[4];     /* staged upload only: host-visible copies of rec, tab, plan, offs */
    int staged;
    VkDescriptorSet set;
} Slab;

enum { DO_UPLOAD = 1, DO_CLEAR = 2, DO_ZERO = 4, DO_SCALES = 8, DO_WEIGHTS = 16, DO_READBACK = 32 };

/* Lay jobs[0..n) out in one batch and create its buffers. Each job gets the offsets of its six regions in
 * the decoded buffer. staged = 0: the four input buffers are host-visible and the shader reads them where
 * the host wrote them. staged = 1: they are device-local, filled by a copy from host-visible buffers. */
static void slab_create(Gpu *g, Job *jobs, uint32_t n, int staged, Slab *s) {
    uint32_t models = 0, i, m;
    uint64_t record_bytes = 0, offsets = 0, out_bytes = 0, sizes[4];
    Buffer *in[4], *bound[6];
    VkDescriptorSetAllocateInfo set_info;
    VkDescriptorBufferInfo buffer_info[6];
    VkWriteDescriptorSet writes[6];

    if (!n || n > MAX_BATCH) DIE("a batch holds 1 to %u records", (unsigned)MAX_BATCH);
    memset(s, 0, sizeof(*s));
    s->staged = staged;
    for (i = 0; i < n; i++) {
        for (m = 0; m < i && jobs[m].model != jobs[i].model; m++) {}
        if (m == i) models++;
        record_bytes += jobs[i].stored_bytes;
        offsets += (uint64_t)jobs[i].view.scale_streams + jobs[i].view.weight_streams + 1u;
        for (m = 0; m < 6u; m++) {   /* six regions per record, each on a 16-byte boundary */
            out_bytes = align16(out_bytes);
            if (m < 3u) { jobs[i].scale_at[m] = (uint32_t)out_bytes; out_bytes += jobs[i].layout.scale_count[m]; }
            else { jobs[i].weight_at[m - 3u] = (uint32_t)out_bytes; out_bytes += jobs[i].layout.symbol_count[m - 3u] / 2u; }
            if (out_bytes > UINT32_MAX - 16u) DIE("the batch is too large for one storage buffer");
        }
        if (jobs[i].stored_bytes & 3u) DIE("a record size is not a multiple of 4");
    }
    out_bytes = align16(out_bytes);
    /* byte offsets are 32-bit in the shader; one storage buffer must fit the device's range limit */
    if (record_bytes > UINT32_MAX || out_bytes > UINT32_MAX || record_bytes > g->limits.maxStorageBufferRange ||
        out_bytes > g->limits.maxStorageBufferRange)
        DIE("%s (maxStorageBufferRange %lu)", n == 1u ? "one record is too large for one storage buffer of this device"
            : "the batch is too large for one storage buffer", (unsigned long)g->limits.maxStorageBufferRange);

    in[0] = &s->rec; in[1] = &s->tab; in[2] = &s->plan; in[3] = &s->offs;
    sizes[0] = record_bytes; sizes[1] = (uint64_t)models * MODEL_SLOT_BYTES;
    sizes[2] = (uint64_t)n * PLAN_WORDS * 4u; sizes[3] = offsets * 4u;
    for (i = 0; i < 4u; i++) {
        if (staged) {
            buffer_create(g, sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, in[i]);
            buffer_create(g, sizes[i], VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &s->stage[i]);
        } else {
            buffer_create(g, sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1, in[i]);
        }
    }
    buffer_create(g, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, &s->out);
    buffer_create(g, (VkDeviceSize)n * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1, &s->err);
    buffer_create(g, out_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 2, &s->back);

    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = g->descriptors;
    set_info.descriptorSetCount = 1u;
    set_info.pSetLayouts = &g->set_layout;
    VK(vkAllocateDescriptorSets(g->device, &set_info, &s->set));
    bound[0] = &s->rec; bound[1] = &s->tab; bound[2] = &s->plan; bound[3] = &s->offs; bound[4] = &s->out; bound[5] = &s->err;
    memset(writes, 0, sizeof(writes));
    for (i = 0; i < 6u; i++) {
        buffer_info[i].buffer = bound[i]->buffer; buffer_info[i].offset = 0; buffer_info[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s->set; writes[i].dstBinding = i; writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_info[i];
    }
    vkUpdateDescriptorSets(g->device, 6u, writes, 0, NULL);
}

static void slab_destroy(Gpu *g, Slab *s) {
    uint32_t i;
    VK(vkResetDescriptorPool(g->device, g->descriptors, 0));
    buffer_destroy(g, &s->rec); buffer_destroy(g, &s->tab); buffer_destroy(g, &s->plan); buffer_destroy(g, &s->offs);
    buffer_destroy(g, &s->out); buffer_destroy(g, &s->err); buffer_destroy(g, &s->back);
    for (i = 0; i < 4u; i++) if (s->staged) buffer_destroy(g, &s->stage[i]);
}

/* The scale symbol of every slot of the three scale CDFs of a validated model blob, one byte each (the
 * table lookup of docs/WNR3-VULKAN.md section 5), built by walking each CDF and checked against the
 * reference binary search on all 2^14 slots. The last table built is kept for the next batch. */
static const unsigned char *scale_table(const unsigned char *blob) {
    static unsigned char built_for[WEX_NR2_MODEL_BYTES], table[SCALE_TABLE_BYTES];
    static int built;
    WexNr2Model model;
    uint32_t m, s, slot;
    if (built && !memcmp(built_for, blob, WEX_NR2_MODEL_BYTES)) return table;
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) DIE("a model blob that was valid is not");
    for (m = 0; m < 3u; m++) {
        const unsigned char *cdf = wex_nr2_scale_cdf(&model, m);
        unsigned char *t = table + (size_t)m * WEX_NR2_TOTAL;
        for (s = 0; s < 256u; s++)
            for (slot = wex_nr_u16(cdf + s * 2u); slot < wex_nr_u16(cdf + s * 2u + 2u); slot++) t[slot] = (unsigned char)s;
        for (slot = 0; slot < WEX_NR2_TOTAL; slot++) {
            uint32_t lo = 0, hi = 256u;
            while (lo + 1u < hi) {
                uint32_t mid = (lo + hi) / 2u;
                if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid;
            }
            if (t[slot] != lo) DIE("the scale table of matrix %u disagrees with the binary search at slot %u", (unsigned)m, (unsigned)slot);
        }
    }
    memcpy(built_for, blob, WEX_NR2_MODEL_BYTES);
    built = 1;
    return table;
}

/* The host's part of the upload: records, model blobs each followed by its scale table, plans and stream
 * offsets into mapped memory. Every word is written byte by byte, little-endian, as the shader reads it. */
static void slab_write(const Gpu *g, const Slab *s, const Job *jobs, uint32_t n) {
    unsigned char *rec = (unsigned char *)(s->staged ? s->stage[0].map : s->rec.map);
    unsigned char *tab = (unsigned char *)(s->staged ? s->stage[1].map : s->tab.map);
    unsigned char *plan = (unsigned char *)(s->staged ? s->stage[2].map : s->plan.map);
    unsigned char *offs = (unsigned char *)(s->staged ? s->stage[3].map : s->offs.map);
    const unsigned char *models[MAX_BATCH];
    uint32_t model_count = 0, i, m, k;
    uint64_t record_bytes = 0, offsets = 0;
    for (i = 0; i < n; i++) {
        const Job *job = &jobs[i];
        unsigned char *p = plan + (size_t)i * PLAN_WORDS * 4u, *o = offs + (size_t)offsets * 4u;
        uint32_t words[PLAN_WORDS], streams = job->view.scale_streams + job->view.weight_streams;
        uint32_t at = (uint32_t)record_bytes + job->view.metadata_bytes, scale_stream = 0, weight_stream = 0, model;
        for (model = 0; model < model_count && models[model] != job->model; model++) {}
        if (model == model_count) {
            models[model_count++] = job->model;
            memcpy(tab + (size_t)model * MODEL_SLOT_BYTES, job->model, WEX_NR2_MODEL_BYTES);
            memcpy(tab + (size_t)model * MODEL_SLOT_BYTES + WEX_NR2_MODEL_BYTES, scale_table(job->model), SCALE_TABLE_BYTES);
        }
        memcpy(rec + (size_t)record_bytes, job->stored, job->stored_bytes);
        memset(words, 0, sizeof(words));
        for (m = 0; m < 3u; m++) {
            words[0u + m] = job->scale_at[m];
            words[3u + m] = job->weight_at[m];
            words[6u + m] = job->layout.scale_count[m];
            words[9u + m] = job->layout.symbol_count[m];
            words[19u + m] = scale_stream;
            words[23u + m] = weight_stream;
            scale_stream += wex_nr2_streams(job->layout.scale_count[m], job->view.scale_chunk);
            weight_stream += wex_nr2_streams(job->layout.symbol_count[m], job->view.weight_chunk);
        }
        words[22] = scale_stream; words[26] = weight_stream;
        if (scale_stream != job->view.scale_streams || weight_stream != job->view.weight_streams) DIE("stream count");
        words[12] = job->view.scale_chunk; words[13] = job->view.weight_chunk;
        words[14] = job->view.model_id; words[15] = job->view.owner; words[16] = job->view.word24;
        words[17] = model * (MODEL_SLOT_BYTES / 4u);
        words[27] = words[17] + WEX_NR2_MODEL_BYTES / 4u;
        words[18] = (uint32_t)offsets;
        for (k = 0; k < PLAN_WORDS; k++) put_u32(p + k * 4u, words[k]);
        for (k = 0; k < streams; k++) {   /* absolute byte offsets in the record buffer */
            put_u32(o + (size_t)k * 4u, at);
            at += wex_nr_u16(job->view.lengths + (size_t)k * 2u);
        }
        put_u32(o + (size_t)streams * 4u, at);
        if (scale_stream > g->limits.maxComputeWorkGroupCount[0] * GROUP ||
            weight_stream > g->limits.maxComputeWorkGroupCount[0] * GROUP) DIE("too many streams for one dispatch");
        record_bytes += job->stored_bytes;
        offsets += streams + 1u;
    }
}

/* One command buffer with the parts named in what, submitted and waited for. DO_UPLOAD copies the staged
 * inputs to the device (nothing when the upload is not staged), DO_CLEAR zeroes the error words, DO_ZERO the
 * decoded buffer, DO_SCALES is pass A for every record, DO_WEIGHTS pass B for every record, DO_READBACK
 * copies the decoded buffer to host-visible memory. With a query pool (2 timestamp queries) the command
 * buffer writes a timestamp before its first command and one after its last. */
static void slab_run(Gpu *g, const Slab *s, const Job *jobs, uint32_t n, int what, VkQueryPool queries) {
    VkCommandBufferAllocateInfo cmd_info;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cmd;
    VkSubmitInfo submit;
    VkFenceCreateInfo fence_info;
    VkFence fence;
    VkBufferCopy copy;
    uint32_t i, m;
    const VkPipelineStageFlags compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkAccessFlags shader = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    memset(&cmd_info, 0, sizeof(cmd_info));
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = g->pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1u;
    VK(vkAllocateCommandBuffers(g->device, &cmd_info, &cmd));
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cmd, &begin));
    if (queries) {
        vkCmdResetQueryPool(cmd, queries, 0, 2u);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    }
    copy.srcOffset = 0; copy.dstOffset = 0;
    if ((what & DO_UPLOAD) && s->staged) {
        const Buffer *in[4];
        in[0] = &s->rec; in[1] = &s->tab; in[2] = &s->plan; in[3] = &s->offs;
        for (i = 0; i < 4u; i++) {
            copy.size = in[i]->bytes;
            vkCmdCopyBuffer(cmd, s->stage[i].buffer, in[i]->buffer, 1u, &copy);
            barrier(cmd, in[i]->buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, transfer, compute);
        }
    }
    if (what & DO_CLEAR) {
        vkCmdFillBuffer(cmd, s->err.buffer, 0, VK_WHOLE_SIZE, 0);
        barrier(cmd, s->err.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, shader, transfer, compute);
    }
    if (what & DO_ZERO) {   /* not needed for the result: a known start for the comparison */
        vkCmdFillBuffer(cmd, s->out.buffer, 0, VK_WHOLE_SIZE, 0);
        barrier(cmd, s->out.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, shader, transfer, compute);
    }
    if (what & (DO_SCALES | DO_WEIGHTS))
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1u, &s->set, 0, NULL);
    for (m = 0; m < 2u; m++) {   /* pass A for every record, then pass B for every record */
        if (!(what & (m ? DO_WEIGHTS : DO_SCALES))) continue;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipeline[2u * g->form + m]);
        for (i = 0; i < n; i++) {
            uint32_t streams = m ? jobs[i].view.weight_streams : jobs[i].view.scale_streams;
            vkCmdPushConstants(cmd, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4u, &i);
            vkCmdDispatch(cmd, (streams + GROUP - 1u) / GROUP, 1u, 1u);
        }
        /* pass B reads the scales pass A wrote; the copy reads what both wrote, the host the error words */
        if (!m) {
            barrier(cmd, s->out.buffer, VK_ACCESS_SHADER_WRITE_BIT, shader, compute, compute);
        } else {
            barrier(cmd, s->out.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, compute, transfer);
            barrier(cmd, s->err.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, compute, VK_PIPELINE_STAGE_HOST_BIT);
        }
    }
    if (what & DO_READBACK) {
        copy.size = s->out.bytes;
        vkCmdCopyBuffer(cmd, s->out.buffer, s->back.buffer, 1u, &copy);
        barrier(cmd, s->back.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, transfer, VK_PIPELINE_STAGE_HOST_BIT);
    }
    if (queries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    VK(vkEndCommandBuffer(cmd));

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK(vkCreateFence(g->device, &fence_info, NULL, &fence));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers = &cmd;
    VK(vkQueueSubmit(g->queue, 1u, &submit, fence));
    VK(vkWaitForFences(g->device, 1u, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(g->device, fence, NULL);
    vkFreeCommandBuffers(g->device, g->pool, 1u, &cmd);
}

/* Decode jobs[0..n) on the device, in one command buffer. Returns the read-back copy of the decoded buffer
 * (caller frees); each job gets its error word and the offsets of its six regions in that copy. */
static unsigned char *gpu_decode(Gpu *g, Job *jobs, uint32_t n) {
    Slab s;
    unsigned char *result;
    uint32_t i;
    slab_create(g, jobs, n, g->staged, &s);
    slab_write(g, &s, jobs, n);
    slab_run(g, &s, jobs, n, DO_UPLOAD | DO_CLEAR | DO_ZERO | DO_SCALES | DO_WEIGHTS | DO_READBACK, VK_NULL_HANDLE);
    for (i = 0; i < n; i++) jobs[i].error = wex_nr_u32((const unsigned char *)s.err.map + (size_t)i * 4u);
    result = (unsigned char *)xmalloc((size_t)s.out.bytes);
    memcpy(result, s.back.map, (size_t)s.out.bytes);
    slab_destroy(g, &s);
    return result;
}

/* ---- comparison with the CPU reference ---------------------------------------------------------------- */

typedef struct {
    unsigned long accepted, refused, different, ceil_blocks;
    uint32_t error_bits;
    uint32_t flag_values;   /* bit f: an accepted record had flag byte f (0..15) */
} Tally;

static size_t layout_bytes(const WexNr2Layout *layout, uint32_t scale_at[3], uint32_t weight_at[3]) {
    size_t at = 0;
    uint32_t m;
    for (m = 0; m < 3u; m++) { scale_at[m] = (uint32_t)at; at += layout->scale_count[m]; }
    for (m = 0; m < 3u; m++) { weight_at[m] = (uint32_t)at; at += layout->symbol_count[m] / 2u; }
    return at;
}

/* Decode the batch on the device and each record with wex_nr2_decode; the verdicts must agree and an accepted
 * record must have the reference's bytes in all six regions. Returns the number of differences. */
static unsigned long run_batch(Gpu *g, Job *jobs, uint32_t n, const char *what, Tally *tally) {
    unsigned char *device = gpu_decode(g, jobs, n);
    unsigned long different = 0;
    uint32_t i, m;
    for (i = 0; i < n; i++) {
        Job *job = &jobs[i];
        uint32_t scale_at[3], weight_at[3], b;
        size_t bytes = layout_bytes(&job->layout, scale_at, weight_at);
        unsigned char *reference = (unsigned char *)xmalloc(bytes), *coded = (unsigned char *)xmalloc(bytes);
        unsigned char *scales[3], *weights[3], *coded_scales[3], *coded_weights[3];
        WexNr2Model model;
        WexNr2Record plain = job->view;
        int cpu, same = 1;
        if (wex_nr2_model_open(job->model, WEX_NR2_MODEL_BYTES, &model)) DIE("a model blob that was valid is not");
        for (m = 0; m < 3u; m++) {
            scales[m] = reference + scale_at[m]; weights[m] = reference + weight_at[m];
            coded_scales[m] = coded + scale_at[m]; coded_weights[m] = coded + weight_at[m];
        }
        cpu = wex_nr2_decode(&job->view, &job->layout, &model, scales, weights);
        if (!cpu != !job->error) {
            same = 0;
            printf("%s record %u: the reference %s it, the device reports error 0x%x\n", what, (unsigned)i,
                   cpu ? "refuses" : "accepts", (unsigned)job->error);
        } else if (!cpu) {
            for (m = 0; m < 3u && same; m++)
                same = !memcmp(device + job->scale_at[m], scales[m], job->layout.scale_count[m]) &&
                       !memcmp(device + job->weight_at[m], weights[m], job->layout.symbol_count[m] / 2u);
            if (!same) {
                printf("%s record %u: accepted by both, the decoded bytes differ\n", what, (unsigned)i);
                for (m = 0; m < 6u; m++) {   /* per region: how many bytes differ, and the first of them */
                    const unsigned char *d = device + (m < 3u ? job->scale_at[m] : job->weight_at[m - 3u]);
                    const unsigned char *r = m < 3u ? scales[m] : weights[m - 3u];
                    uint32_t size = m < 3u ? job->layout.scale_count[m] : job->layout.symbol_count[m - 3u] / 2u, first = size, n_bad = 0;
                    for (b = 0; b < size; b++) if (d[b] != r[b]) { if (!n_bad++) first = b; }
                    if (n_bad) printf("  %s of matrix %u: %u of %u bytes differ, the first at %u (device 0x%02x, reference 0x%02x)\n",
                                      m < 3u ? "scales" : "weights", (unsigned)(m % 3u), (unsigned)n_bad, (unsigned)size,
                                      (unsigned)first, (unsigned)d[first], (unsigned)r[first]);
                }
            }
            /* how many blocks the ceil-form inverse changed: the same streams decoded without it */
            plain.flags &= (unsigned char)~WEX_NR3_FLAG_CEIL_MASK;
            if (job->view.flags & WEX_NR3_FLAG_CEIL_MASK) {
                if (wex_nr2_decode(&plain, &job->layout, &model, coded_scales, coded_weights)) DIE("coded decode");
                for (m = 0; m < 3u; m++)
                    for (b = 0; b < job->layout.scale_count[m]; b++)
                        tally->ceil_blocks += coded_scales[m][b] != scales[m][b];
            }
        }
        if (same && cpu) { tally->refused++; tally->error_bits |= job->error; }
        if (same && !cpu) { tally->accepted++; tally->flag_values |= 1u << (job->view.flags & 15u); }
        if (!same) { tally->different++; different++; }
        free(coded); free(reference);
    }
    free(device);
    return different;
}

/* ---- containers --------------------------------------------------------------------------------------- */

typedef struct {
    const char *path;
    FILE *f;
    unsigned char *index;      /* the index as stored, validated by wex4_open */
    unsigned char **models;    /* per blob: the bytes of a WNM2 blob, read when a record first names it */
    Wex4View view;
    uint32_t records, blobs;
} Container;

static void read_at(const Container *c, uint64_t offset, unsigned char *to, size_t bytes) {
    if (seek_file(c->f, offset)) DIE("%s: cannot seek", c->path);
    if (fread(to, 1u, bytes, c->f) != bytes) DIE("%s: short read", c->path);
}

static void container_open(const char *path, Container *c) {
    static unsigned char page[WEX4_PAGE];
    Wex4Header h;
    int64_t size;
    memset(c, 0, sizeof(*c));
    c->path = path;
    c->f = fopen(path, "rb");
    if (!c->f) DIE("%s: cannot open", path);
    if (seek_end(c->f) || (size = tell_file(c->f)) < 0) DIE("%s: cannot seek", path);
    if (size < (int64_t)WEX4_PAGE) DIE("%s: does not open as WEX4", path);
    read_at(c, 0, page, WEX4_PAGE);
    if (wex4_parse_header(page, (uint64_t)size, &h)) DIE("%s: does not open as WEX4", path);
    c->index = (unsigned char *)xmalloc((size_t)wex4_align(h.index_bytes));
    read_at(c, h.index_offset, c->index, (size_t)wex4_align(h.index_bytes));
    if (wex4_open(&h, c->index, (size_t)wex4_align(h.index_bytes), &c->view)) DIE("%s: does not open as WEX4", path);
    c->records = h.records; c->blobs = h.blobs;
    c->models = (unsigned char **)xmalloc((size_t)h.blobs * sizeof(*c->models));
}

static void container_close(Container *c) {
    uint32_t b;
    for (b = 0; b < c->blobs; b++) free(c->models[b]);
    free(c->models); free(c->index);
    fclose(c->f);
}

/* The bytes of blob b when it has the size of a WNM2 model, else NULL. Kept until the container closes. */
static const unsigned char *container_model(Container *c, uint32_t b) {
    Wex4Blob blob;
    if (b >= c->blobs) return NULL;
    wex4_get_blob(&c->view, b, &blob);
    if (blob.bytes != WEX_NR2_MODEL_BYTES) return NULL;
    if (!c->models[b]) {
        c->models[b] = (unsigned char *)xmalloc(WEX_NR2_MODEL_BYTES);
        read_at(c, blob.offset, c->models[b], WEX_NR2_MODEL_BYTES);
    }
    return c->models[b];
}

/* The host-side checks of wex4_decode_record (c/wex4.h) for record r, whose rec.stored bytes are at stored.
 * 0: job is ready for the device; 1: not a WNR3 record; -1: refused on the host. */
static int container_job(Container *c, uint32_t r, const unsigned char *stored, Job *job) {
    Wex4Record rec;
    Wex4Blob blob;
    WexEntry entry;
    WexNr2Model model;
    const unsigned char *bytes;
    wex4_get_record(&c->view, r, &rec);
    if (rec.codec != WEX4_CODEC_WNR3) return 1;
    if (wex4_expert_entry(&c->view, r, &entry)) return -1;
    wex4_get_blob(&c->view, rec.blob - 1u, &blob);
    bytes = container_model(c, rec.blob - 1u);
    if (!bytes || wex_nr2_model_open(bytes, WEX_NR2_MODEL_BYTES, &model) ||
        model.id != blob.id || wex_nr_u32(stored + 28u) != rec.owner ||
        !(stored[27] & WEX_NR3_FLAG_CZ) != !(rec.flags & WEX4_REC_VALUE_EXACT)) return -1;
    memset(job, 0, sizeof(*job));
    wex_nr2_layout(&entry, &job->layout);
    if (wex_nrx_parse(stored, rec.stored, &job->layout, model.id, WEX_NR3_MAGIC, &job->view)) return -1;
    job->stored = stored; job->stored_bytes = rec.stored; job->model = bytes;
    return 0;
}

static uint64_t rng_state;
static uint32_t rng(void) {   /* xorshift64* */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * UINT64_C(2685821657736338717)) >> 32);
}

/* The records waiting for the device, and their stored bytes (owned here, freed after the batch ran). */
typedef struct {
    Job jobs[MAX_BATCH];
    unsigned char *bytes[MAX_BATCH];
    uint32_t n;
    uint64_t stored, decoded, limit;
    unsigned long batches, failures;
} Batch;

static void batch_run(Gpu *g, Batch *b, const char *what, Tally *tally) {
    uint32_t i;
    if (!b->n) return;
    b->failures += run_batch(g, b->jobs, b->n, what, tally);
    for (i = 0; i < b->n; i++) { free(b->bytes[i]); b->bytes[i] = NULL; }
    b->n = 0; b->stored = 0; b->decoded = 0;
    b->batches++;
}

/* job reads the record at bytes; the batch takes both. The batch runs first when the record does not fit. */
static void batch_add(Gpu *g, Batch *b, const Job *job, unsigned char *bytes, const char *what, Tally *tally) {
    uint64_t decoded = job_decoded_bytes(job);
    if (b->n == MAX_BATCH || b->stored + job->stored_bytes > b->limit || b->decoded + decoded > b->limit)
        batch_run(g, b, what, tally);   /* a record above the limit gets a batch of its own */
    b->jobs[b->n] = *job; b->bytes[b->n] = bytes;
    b->stored += job->stored_bytes; b->decoded += decoded;
    b->n++;
}

/* mode 0 decode, 1 reject, 2 mutate (copies per record) */
static unsigned long run_file(Gpu *g, const char *path, int mode, uint32_t copies) {
    static Batch batch;
    const char *every_text = getenv("WNR3_VK_EVERY");
    Container c;
    Tally tally;
    unsigned long wnr3 = 0, taken = 0, host_refused = 0, failures = 0;
    uint32_t r, k, every = every_text && every_text[0] ? (uint32_t)strtoul(every_text, NULL, 10) : 1u;
    if (!every) DIE("WNR3_VK_EVERY must be at least 1");
    memset(&tally, 0, sizeof(tally));
    memset(&batch, 0, sizeof(batch));
    batch.limit = batch_limit(g);
    container_open(path, &c);
    for (r = 0; r < c.records; r++) {
        Wex4Record rec;
        unsigned char *original;
        Job job;
        uint32_t used = 32u, position;
        int original_status;
        wex4_get_record(&c.view, r, &rec);
        if (rec.codec != WEX4_CODEC_WNR3) continue;
        if (wnr3++ % every) continue;
        taken++;
        original = (unsigned char *)xmalloc(rec.stored);
        read_at(&c, rec.offset, original, rec.stored);
        original_status = container_job(&c, r, original, &job);
        if (!original_status) used = job.view.metadata_bytes + job.view.scale_payload_bytes + job.view.weight_payload_bytes;
        for (k = 0; k < (mode == 2 ? copies : 1u); k++) {
            unsigned char *stored = original;
            int status = original_status;
            if (mode == 2) {   /* one byte of the used part of the record gets another value */
                stored = (unsigned char *)xmalloc(rec.stored);
                memcpy(stored, original, rec.stored);
                position = rng() % used;
                stored[position] ^= (unsigned char)(1u + rng() % 255u);
                status = container_job(&c, r, stored, &job);
            }
            if (status < 0) {
                /* refused before any decode; the reference path refuses it too (the same checks) */
                unsigned char *out = (unsigned char *)xmalloc(rec.decoded);
                const unsigned char *bytes = container_model(&c, rec.blob - 1u);
                WexNr2Model model;
                int cpu = WEX4_E_DATA;
                if (bytes && !wex_nr2_model_open(bytes, WEX_NR2_MODEL_BYTES, &model))
                    cpu = wex4_decode_record(&c.view, r, stored, out, &model, 0);
                if (!cpu) { printf("%s record %u: refused by the host checks, accepted by wex4_decode_record\n", path, (unsigned)r); failures++; }
                host_refused++;
                free(out);
                if (mode == 2) free(stored);
                continue;
            }
            batch_add(g, &batch, &job, stored, path, &tally);
            if (mode != 2) original = NULL;   /* the batch owns it now */
        }
        free(original);
    }
    batch_run(g, &batch, path, &tally);
    failures += batch.failures;
    printf("%s: %lu WNR3 records, %lu taken%s, %lu batches of at most %lu bytes; device and reference agree on %lu accepted "
           "(flag bytes:", path, wnr3, taken, mode == 2 ? " and mutated" : "", batch.batches, (unsigned long)batch.limit,
           tally.accepted);
    for (k = 0; k < 16u; k++) if (tally.flag_values >> k & 1u) printf(" 0x%02x", (unsigned)k);
    printf("; %lu blocks undone by the ceil-form inverse) and %lu refused (error bits 0x%x); %lu refused by the host checks; "
           "%lu differences\n", tally.ceil_blocks, tally.refused, (unsigned)tally.error_bits, host_refused, tally.different);
    if (!wnr3) { printf("%s: no WNR3 record\n", path); failures++; }
    if (mode == 0 && (tally.refused || host_refused)) { printf("%s: a record was refused\n", path); failures++; }
    if (mode == 1 && !(tally.refused + host_refused)) { printf("%s: no record was refused\n", path); failures++; }
    container_close(&c);
    return failures;
}

/* ---- timing ------------------------------------------------------------------------------------------- */

static int by_value(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of(double *t, uint32_t runs) {   /* sorts t */
    qsort(t, runs, sizeof(*t), by_value);
    return runs & 1u ? t[runs / 2u] : (t[runs / 2u - 1u] + t[runs / 2u]) / 2.0;
}

/* median, minimum and maximum of t[0..runs), each the time of one pass over the batch */
static void report(const char *name, double *t, uint32_t runs, uint32_t records, uint64_t decoded) {
    double median = median_of(t, runs);
    printf("bench: %-44s %u runs, median %9.3f ms per batch (min %9.3f, max %9.3f), %8.3f ms per record, %8.1f MB/s\n",
           name, (unsigned)runs, median, t[0], t[runs - 1u], median / records, (double)decoded / 1e6 / (median / 1e3));
}

/* the device times of the two command buffers that wrote the two query pools, in ms */
static void read_stamps(const Gpu *g, const VkQueryPool queries[2], double ms[2]) {
    uint64_t stamp[2], mask = g->timestamp_bits >= 64u ? ~UINT64_C(0) : (UINT64_C(1) << g->timestamp_bits) - 1u;
    uint32_t m;
    for (m = 0; m < 2u; m++) {   /* ticks of timestampPeriod ns, timestamp_bits of them valid */
        VK(vkGetQueryPoolResults(g->device, queries[m], 0, 2u, sizeof(stamp), stamp, sizeof(stamp[0]),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        ms[m] = (double)((stamp[1] - stamp[0]) & mask) * (double)g->limits.timestampPeriod * 1e-6;
    }
}

/* 1 when one of the six regions of job in the read-back copy device differs from the reference's bytes,
 * whose regions start at scale_at and weight_at */
static int regions_differ(const unsigned char *device, const Job *job, const unsigned char *reference,
                          const uint32_t scale_at[3], const uint32_t weight_at[3]) {
    uint32_t m;
    for (m = 0; m < 3u; m++)
        if (memcmp(device + job->scale_at[m], reference + scale_at[m], job->layout.scale_count[m]) ||
            memcmp(device + job->weight_at[m], reference + weight_at[m], job->layout.symbol_count[m] / 2u)) return 1;
    return 0;
}

/* the error words of the batch after a run: each nonzero one is a failure of the bench */
static unsigned long bench_errors(const Slab *s, uint32_t n, uint32_t run) {
    unsigned long failures = 0;
    uint32_t i;
    for (i = 0; i < n; i++)
        if (wex_nr_u32((const unsigned char *)s->err.map + (size_t)i * 4u)) {
            printf("bench: run %u, record %u: error word 0x%x\n", (unsigned)run, (unsigned)i,
                   (unsigned)wex_nr_u32((const unsigned char *)s->err.map + (size_t)i * 4u));
            failures++;
        }
    return failures;
}

/* The first `want` WNR3 records of a file as one batch, decoded `runs` times on the device after one second
 * of decode passes and one whole pass that are not timed, in four parts timed on the host clock:
 *   upload     the host writes records, model blobs, plans and stream offsets into mapped buffers; with a
 *              staged upload also the copy of them to device-local buffers (submit and wait)
 *   pass A     the error words cleared and every scale stream decoded (one submit and wait)
 *   pass B     every weight stream decoded (one submit and wait)
 *   read-back  the decoded buffer copied to host-visible memory and from there to host memory; an engine
 *              that keeps the bytes on the device does not do this
 * Pass A and pass B are also timed on the device, by timestamps written before the first and after the
 * last command of their command buffer (when the queue has timestamps): no submit and no wait in them.
 * The buffers, the pipelines and the descriptor set are created before the first pass and are in no part.
 * Then the same records on the CPU: the reference decoder on one thread (at most 5 passes) and, in a build
 * with -DWNR3_VK_FAST, the fast decoder of wex4/research/fast-cpu-decode on one thread and on all threads,
 * its tables built before the clock starts. The error words of every device run are checked; the device's
 * bytes are compared with the reference's once per form, in one more run after the timed ones (from a zeroed
 * buffer, not timed); the fast decoder's output is compared with the reference's in every run. With
 * WNR3_VK_BENCH_CPU=0 the CPU decoders are not timed: the reference decodes the batch once, for the
 * comparison. With a second form of the shaders (WNR3_VK_SHADERS_B), RUNS pairs of the two forms follow the
 * timed runs: device timestamps of pass A and pass B of each form, the order inside a pair alternating
 * (A B, B A, ...), and per pair the ratio of form B to form A. */
static unsigned long bench(Gpu *g, const char *path, uint32_t runs, uint32_t want) {
    static Job jobs[MAX_BATCH];
    enum { T_UPLOAD, T_SCALES, T_WEIGHTS, T_DECODE, T_BACK, T_NEEDED, T_GPU_SCALES, T_GPU_WEIGHTS, T_GPU_DECODE,
           T_REFERENCE, T_FAST1, T_FASTN, T_COUNT };
    const char *cpu_text = getenv("WNR3_VK_BENCH_CPU");
    int cpu = !cpu_text || cpu_text[0] != '0';
    unsigned char *bytes[MAX_BATCH], *reference[MAX_BATCH], *device, *device_b = NULL;
    unsigned char *scales[3], *weights[3];
    WexNr2Model model;
    Container c;
    Slab s;
    VkQueryPool queries[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    double *t[T_COUNT];
    uint64_t stored = 0, decoded = 0, limit = batch_limit(g);
    uint32_t n = 0, r, i, m, k, cpu_runs = !cpu ? 1u : runs < 5u ? runs : 5u, scale_at[3], weight_at[3];
    unsigned long failures = 0;

    container_open(path, &c);
    for (r = 0; r < c.records && n < want && n < MAX_BATCH; r++) {
        Wex4Record rec;
        wex4_get_record(&c.view, r, &rec);
        if (rec.codec != WEX4_CODEC_WNR3) continue;
        bytes[n] = (unsigned char *)xmalloc(rec.stored);
        read_at(&c, rec.offset, bytes[n], rec.stored);
        if (container_job(&c, r, bytes[n], &jobs[n])) DIE("%s: record %u is refused by the host checks", path, (unsigned)r);
        if (stored + rec.stored > limit || decoded + job_decoded_bytes(&jobs[n]) > limit) { free(bytes[n]); break; }
        stored += rec.stored; decoded += job_decoded_bytes(&jobs[n]);
        n++;
    }
    if (!n) DIE("%s: no WNR3 record fits a batch", path);
    for (k = 0; k < (uint32_t)T_COUNT; k++) t[k] = (double *)xmalloc((size_t)runs * sizeof(double));
    slab_create(g, jobs, n, g->staged, &s);
    device = (unsigned char *)xmalloc((size_t)s.out.bytes);
    printf("bench: %s: %u records in one batch, %lu stored bytes, %lu decoded bytes; the first record has %u + %u streams, "
           "chunks %u:%u, flag byte 0x%02x\n", path, (unsigned)n, (unsigned long)stored, (unsigned long)decoded,
           (unsigned)jobs[0].view.scale_streams, (unsigned)jobs[0].view.weight_streams, (unsigned)jobs[0].view.weight_chunk,
           (unsigned)jobs[0].view.scale_chunk, (unsigned)jobs[0].view.flags);
    printf("bench: %s upload; memory property flags: records 0x%x, decoded 0x%x, read-back 0x%x "
           "(1 device-local, 2 host-visible, 4 coherent, 8 cached); pipelines created in %.1f ms\n",
           g->staged ? "staged" : "direct", (unsigned)s.rec.flags, (unsigned)s.out.flags, (unsigned)s.back.flags, g->pipeline_ms);

    if (g->timestamp_bits) {
        VkQueryPoolCreateInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = 2u;
        for (k = 0; k < 2u; k++) VK(vkCreateQueryPool(g->device, &info, NULL, &queries[k]));
    } else {
        printf("bench: the queue has no timestamps: no device times\n");
    }
    {   /* warm-up, not timed: decode passes for one second, so that a GPU has left its idle clocks */
        double start = now_ms();
        slab_write(g, &s, jobs, n);
        slab_run(g, &s, jobs, n, DO_UPLOAD, VK_NULL_HANDLE);
        while (now_ms() - start < 1000.0) slab_run(g, &s, jobs, n, DO_CLEAR | DO_SCALES | DO_WEIGHTS, VK_NULL_HANDLE);
    }
    for (k = 0; k <= runs; k++) {   /* pass 0 is not timed */
        double t0 = now_ms(), t1, t2, t3, t4, gpu[2] = {0, 0};
        slab_write(g, &s, jobs, n);
        if (g->staged) slab_run(g, &s, jobs, n, DO_UPLOAD, VK_NULL_HANDLE);
        t1 = now_ms();
        slab_run(g, &s, jobs, n, DO_CLEAR | DO_SCALES, queries[0]);
        t2 = now_ms();
        slab_run(g, &s, jobs, n, DO_WEIGHTS, queries[1]);
        t3 = now_ms();
        slab_run(g, &s, jobs, n, DO_READBACK, VK_NULL_HANDLE);
        memcpy(device, s.back.map, (size_t)s.out.bytes);
        t4 = now_ms();
        if (queries[0]) read_stamps(g, queries, gpu);
        failures += bench_errors(&s, n, k);
        if (!k) continue;
        t[T_UPLOAD][k - 1u] = t1 - t0; t[T_SCALES][k - 1u] = t2 - t1; t[T_WEIGHTS][k - 1u] = t3 - t2;
        t[T_DECODE][k - 1u] = t3 - t1; t[T_BACK][k - 1u] = t4 - t3; t[T_NEEDED][k - 1u] = t3 - t0;
        t[T_GPU_SCALES][k - 1u] = gpu[0]; t[T_GPU_WEIGHTS][k - 1u] = gpu[1]; t[T_GPU_DECODE][k - 1u] = gpu[0] + gpu[1];
    }
    if (g->forms == 2u && queries[0]) {
        /* the second form of the shader (WNR3_VK_SHADERS_B) against the first, in pairs on the batch as it was
         * uploaded: A B, B A, A B, ... Device times of pass A and of pass B only; per pair the ratio B / A */
        double *pt[2][3], *ratio[3], gpu[2];
        static const char *const part[3] = {"pass A", "pass B", "pass A + pass B"};
        char name[64];
        uint32_t f, x;
        for (x = 0; x < 3u; x++) {
            for (f = 0; f < 2u; f++) pt[f][x] = (double *)xmalloc((size_t)runs * sizeof(double));
            ratio[x] = (double *)xmalloc((size_t)runs * sizeof(double));
        }
        for (k = 0; k < runs; k++) {
            for (x = 0; x < 2u; x++) {
                g->form = f = (k & 1u) ^ x;
                slab_run(g, &s, jobs, n, DO_CLEAR | DO_SCALES, queries[0]);
                slab_run(g, &s, jobs, n, DO_WEIGHTS, queries[1]);
                read_stamps(g, queries, gpu);
                failures += bench_errors(&s, n, k);
                pt[f][0][k] = gpu[0]; pt[f][1][k] = gpu[1]; pt[f][2][k] = gpu[0] + gpu[1];
            }
            for (x = 0; x < 3u; x++) ratio[x][k] = pt[1][x][k] / pt[0][x][k];
        }
        g->form = 0;
        for (x = 0; x < 3u; x++) {
            double median = median_of(ratio[x], runs);   /* sorts the ratios: min and max are read after it */
            for (f = 0; f < 2u; f++) {
                snprintf(name, sizeof(name), "pairs, form %c, device: %s", f ? 'B' : 'A', part[x]);
                report(name, pt[f][x], runs, n, decoded);
            }
            printf("bench: pairs, form B / form A, %-25s %u pairs, median %.3f (min %.3f, max %.3f)\n", part[x],
                   (unsigned)runs, median, ratio[x][0], ratio[x][runs - 1u]);
            for (f = 0; f < 2u; f++) free(pt[f][x]);
            free(ratio[x]);
        }
    }
    for (k = 0; k < 2u; k++) if (queries[k]) vkDestroyQueryPool(g->device, queries[k], NULL);
    /* once more from a zeroed buffer, for the comparison (each form) */
    slab_write(g, &s, jobs, n);
    slab_run(g, &s, jobs, n, DO_UPLOAD, VK_NULL_HANDLE);
    if (g->forms == 2u) {
        device_b = (unsigned char *)xmalloc((size_t)s.out.bytes);
        g->form = 1;
        slab_run(g, &s, jobs, n, DO_CLEAR | DO_ZERO | DO_SCALES | DO_WEIGHTS | DO_READBACK, VK_NULL_HANDLE);
        memcpy(device_b, s.back.map, (size_t)s.out.bytes);
        failures += bench_errors(&s, n, runs);
        g->form = 0;
    }
    slab_run(g, &s, jobs, n, DO_CLEAR | DO_ZERO | DO_SCALES | DO_WEIGHTS | DO_READBACK, VK_NULL_HANDLE);
    memcpy(device, s.back.map, (size_t)s.out.bytes);

    for (i = 0; i < n; i++) reference[i] = (unsigned char *)xmalloc(layout_bytes(&jobs[i].layout, scale_at, weight_at));
    for (k = 0; k < cpu_runs; k++) {
        double t0 = now_ms();
        for (i = 0; i < n; i++) {
            layout_bytes(&jobs[i].layout, scale_at, weight_at);
            for (m = 0; m < 3u; m++) { scales[m] = reference[i] + scale_at[m]; weights[m] = reference[i] + weight_at[m]; }
            if (wex_nr2_model_open(jobs[i].model, WEX_NR2_MODEL_BYTES, &model) ||
                wex_nr2_decode(&jobs[i].view, &jobs[i].layout, &model, scales, weights))
                DIE("%s: the reference refuses record %u of the batch", path, (unsigned)i);
        }
        t[T_REFERENCE][k] = now_ms() - t0;
    }
    for (i = 0; i < n; i++) {
        layout_bytes(&jobs[i].layout, scale_at, weight_at);
        if (wex_nr_u32((const unsigned char *)s.err.map + (size_t)i * 4u) ||
            regions_differ(device, &jobs[i], reference[i], scale_at, weight_at)) {
            printf("bench: record %u: the device and the reference differ\n", (unsigned)i); failures++;
        }
        if (device_b && regions_differ(device_b, &jobs[i], reference[i], scale_at, weight_at)) {
            printf("bench: record %u: form B on the device and the reference differ\n", (unsigned)i); failures++;
        }
    }
    free(device_b);
#ifdef WNR3_VK_FAST
    if (cpu) {
        static WexFastModel tables;
        static const WexFastOptions one = {1, 1, 4, 1}, all = {0, 1, 4, 1};   /* the header's default kernel */
        unsigned char *fast = NULL;
        int pass;
        for (pass = 0; pass < 2; pass++) {
            for (k = 0; k <= runs; k++) {   /* pass 0 is not timed */
                double total = 0, t0;
                for (i = 0; i < n; i++) {
                    size_t size = layout_bytes(&jobs[i].layout, scale_at, weight_at);
                    fast = (unsigned char *)realloc(fast, size);
                    if (!fast) DIE("out of memory");
                    for (m = 0; m < 3u; m++) { scales[m] = fast + scale_at[m]; weights[m] = fast + weight_at[m]; }
                    if (wex_nr2_model_open(jobs[i].model, WEX_NR2_MODEL_BYTES, &model)) DIE("model");
                    if (tables.id != model.id || (!i && !k && !pass)) wex_fast_model_init(&tables, &model);
                    t0 = now_ms();
                    if (wex_fast_decode_with(&tables, &jobs[i].view, &jobs[i].layout, &model, scales, weights, pass ? &all : &one))
                        DIE("%s: the fast decoder refuses record %u of the batch", path, (unsigned)i);
                    total += now_ms() - t0;
                    if (memcmp(fast, reference[i], size)) {
                        printf("bench: record %u: the fast decoder and the reference differ\n", (unsigned)i); failures++;
                    }
                }
                if (k) t[pass ? T_FASTN : T_FAST1][k - 1u] = total;
            }
        }
        free(fast);
    }
#endif
    report(g->staged ? "device: upload (host writes, staged copy)" : "device: upload (host writes)", t[T_UPLOAD], runs, n, decoded);
    report("device: pass A, scales", t[T_SCALES], runs, n, decoded);
    report("device: pass B, weights", t[T_WEIGHTS], runs, n, decoded);
    report("device: decode (pass A + pass B)", t[T_DECODE], runs, n, decoded);
    report("device: upload + decode", t[T_NEEDED], runs, n, decoded);
    report("device: read-back (comparison only)", t[T_BACK], runs, n, decoded);
    if (queries[0]) {
        report("device timestamps: pass A", t[T_GPU_SCALES], runs, n, decoded);
        report("device timestamps: pass B", t[T_GPU_WEIGHTS], runs, n, decoded);
        report("device timestamps: pass A + pass B", t[T_GPU_DECODE], runs, n, decoded);
    }
    if (cpu) report("CPU: reference decoder, 1 thread", t[T_REFERENCE], cpu_runs, n, decoded);
    else printf("bench: CPU decoders not timed (WNR3_VK_BENCH_CPU=0)\n");
#ifdef WNR3_VK_FAST
    if (cpu) {
        char name[64];
        report("CPU: fast decoder (" WEXF_KERNELS "), 1 thread", t[T_FAST1], runs, n, decoded);
#ifdef _OPENMP
        snprintf(name, sizeof(name), "CPU: fast decoder (" WEXF_KERNELS "), %d threads", omp_get_max_threads());
#else
        snprintf(name, sizeof(name), "CPU: fast decoder (" WEXF_KERNELS "), built without OpenMP");
#endif
        report(name, t[T_FASTN], runs, n, decoded);
    }
#endif
    printf("bench: %lu differences from the reference\n", failures);
    for (i = 0; i < n; i++) { free(reference[i]); free(bytes[i]); }
    for (k = 0; k < (uint32_t)T_COUNT; k++) free(t[k]);
    free(device);
    slab_destroy(g, &s);
    container_close(&c);
    return failures;
}

/* ---- synthetic records -------------------------------------------------------------------------------- */

enum {
    B_NONE,
    B_CEIL,     /* a coded scale 0 in a max-5 block of a ceil-form matrix */
    B_CZ,       /* a nibble 8 in a canonical-zero record */
    B_BYTE,     /* one payload byte changed after encoding: a wrong hash or a stream that misses its end */
    B_OWNER,    /* the owner word changed after encoding: every stream decodes, every hash is wrong */
    B_TAIL,     /* a scale stream with one byte after its last symbol */
    B_LOW,      /* a scale stream that opens below L and, the range check apart, decodes to the same bytes */
    B_HIGH      /* the same with an initial state of 256 L or more */
};

typedef struct {
    const char *name;
    uint32_t rows[3];          /* scales of matrix m */
    uint32_t weight_chunk, scale_chunk;
    int cz;                    /* canonical zero: no nibble 8 in the data, flag CZ */
    int ceil;                  /* 0 plain data; 1 ceil-form data (some blocks with coded max 5, no source max 5) */
    int broken;                /* B_NONE: valid */
} Case;

static uint32_t be32(const unsigned char *p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
static void put_be32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v >> 24); p[1] = (unsigned char)(v >> 16); p[2] = (unsigned char)(v >> 8); p[3] = (unsigned char)v;
}

/* Rewrite one scale stream of an encoded, unpadded record (capacity: 1 byte more) so that the reference
 * refuses the record for the stream's structure only: a decoder without that one check would decode the
 * reference's bytes and find the right hash.
 *   B_TAIL  one byte is appended to the stream.
 *   B_LOW   the first symbol is encoded again from the state shifted right by 8 bits; the byte shifted out
 *           becomes the first payload byte. The initial state is then below L; a decoder that does not look
 *           reads that byte back and goes on as before.
 *   B_HIGH  the first symbol is encoded again from the state after its one renormalisation byte, which is
 *           dropped. The initial state is then 2^31 or more; a decoder that does not look needs no byte
 *           after the first symbol and goes on as before. Needs a stream whose first step read exactly one
 *           byte into a state below 2^18 * frequency: the first such stream is taken. */
static void break_scale_stream(unsigned char *record, size_t *written, const WexNr2Model *model, const WexNr2Layout *layout,
                               uint32_t weight_chunk, uint32_t scale_chunk, int how) {
    uint32_t scale_streams = 0, streams = 0, m, g = 0, at;
    for (m = 0; m < 3u; m++) {
        scale_streams += wex_nr2_streams(layout->scale_count[m], scale_chunk);
        streams += wex_nr2_streams(layout->symbol_count[m], weight_chunk);
    }
    streams += scale_streams;
    at = WEX_NR2_HEADER_BYTES + 2u * streams;
    for (m = 0; m < 3u; m++) {
        const unsigned char *cdf = wex_nr2_scale_cdf(model, m);
        uint32_t first;
        for (first = 0; first < layout->scale_count[m]; first += scale_chunk, g++) {
            unsigned char *lengths = record + WEX_NR2_HEADER_BYTES + (size_t)g * 2u, *p = record + at;
            uint32_t length = wex_nr_u16(lengths), state = be32(p), slot = state & (WEX_NR2_TOTAL - 1u);
            uint32_t symbol = 0, start, frequency, read = 0, grown, shrunk = 0;
            int32_t change;
            while (wex_nr_u16(cdf + symbol * 2u + 2u) <= slot) symbol++;
            start = wex_nr_u16(cdf + symbol * 2u); frequency = wex_nr_u16(cdf + symbol * 2u + 2u) - start;
            state = frequency * (state >> WEX_NR2_PRECISION) + slot - start;      /* after the first symbol */
            grown = state;
            while (grown < WEX_NR_L && 4u + read < length) grown = grown << 8 | p[4u + read++];
            if (how == B_HIGH && (read != 1u || grown / frequency < (1u << 17) || grown / frequency >= (1u << 18))) {
                at += length;
                continue;
            }
            if (how == B_TAIL) {
                change = 1;
                memmove(p + length + 1u, p + length, *written - at - length);
                p[length] = 0x5a;
            } else if (how == B_LOW) {
                change = 1;
                memmove(p + 5u, p + 4u, *written - at - 4u);
                p[4] = (unsigned char)state;
                state >>= 8;
                shrunk = ((state / frequency) << WEX_NR2_PRECISION) + state % frequency + start;
                if (shrunk >= WEX_NR_L) DIE("selftest: the low initial state is not low");
                put_be32(p, shrunk);
            } else {
                change = -1;
                memmove(p + 4u, p + 5u, *written - at - 5u);
                record[*written - 1u] = 0;
                shrunk = ((grown / frequency) << WEX_NR2_PRECISION) + grown % frequency + start;
                if (shrunk < (WEX_NR_L << 8)) DIE("selftest: the high initial state is not high");
                put_be32(p, shrunk);
            }
            length = (uint32_t)((int32_t)length + change);
            if (length > 65535u) DIE("selftest: a stream is too long to grow");
            lengths[0] = (unsigned char)length; lengths[1] = (unsigned char)(length >> 8);
            put_u32(record + 8u, (uint32_t)((int32_t)wex_nr_u32(record + 8u) + change));   /* scale payload bytes */
            *written = (size_t)((long)*written + change);
            return;
        }
    }
    DIE("selftest: no scale stream fits the broken case %d", how);
}

/* only < 0: all twelve records in one batch. Else record `only` alone (for a look at one case; the counts of
 * the whole test are then not checked). */
static unsigned long selftest(Gpu *g, int only) {
    static const Case cases[] = {
        {"plain, smallest chunks, odd scale counts", {37u, 1030u, 513u}, 512u, 256u, 0, 0, B_NONE},
        {"plain, largest chunks", {1200u, 700u, 3u}, 16384u, 16384u, 0, 0, B_NONE},
        {"canonical zero", {64u, 65u, 66u}, 2048u, 512u, 1, 0, B_NONE},
        {"ceil form", {500u, 301u, 1u}, 2048u, 512u, 0, 1, B_NONE},
        {"ceil form and canonical zero, chunk 4096", {259u, 1026u, 130u}, 4096u, 1024u, 1, 1, B_NONE},
        {"ceil form with a coded scale 0 in a max-5 block", {70u, 80u, 90u}, 1024u, 256u, 0, 1, B_CEIL},
        {"nibble 8 in a canonical-zero record", {70u, 80u, 90u}, 1024u, 256u, 1, 0, B_CZ},
        {"one payload byte changed", {70u, 80u, 90u}, 1024u, 256u, 0, 1, B_BYTE},
        {"owner changed: every stream has a wrong hash and nothing else", {70u, 80u, 90u}, 1024u, 256u, 0, 0, B_OWNER},
        {"a scale stream one byte longer than its symbols", {2100u, 900u, 300u}, 2048u, 256u, 0, 0, B_TAIL},
        {"a scale stream with an initial state below L", {2100u, 900u, 300u}, 2048u, 256u, 0, 0, B_LOW},
        {"a scale stream with an initial state of 256 L or more", {8000u, 4000u, 2000u}, 2048u, 256u, 0, 0, B_HIGH},
    };
    enum { CASES = (int)(sizeof(cases) / sizeof(cases[0])), VALID = 5, BROKEN = 7 };
    static uint64_t weight_counts[3u * WEX_NR2_CONTEXTS * 16u], scale_counts[3u * 256u];
    static unsigned char blob[WEX_NR2_MODEL_BYTES];
    WexNr2Model model;
    Job jobs[CASES];
    unsigned char *records[CASES];
    unsigned long failures = 0;
    Tally tally;
    uint32_t c, m, i, from = 0, to = (uint32_t)CASES;

    /* a model with uneven frequencies, so that streams are not all the same length */
    rng_state = UINT64_C(0x5741524e33564b31);
    for (i = 0; i < 3u * WEX_NR2_CONTEXTS * 16u; i++) weight_counts[i] = 1u + (rng() & 15u) * (rng() & 15u);
    for (i = 0; i < 3u * 256u; i++) scale_counts[i] = 1u + (rng() & 31u);
    wex_nr2_model_build(weight_counts, scale_counts, blob);
    if (wex_nr2_model_open(blob, sizeof(blob), &model)) DIE("selftest: the synthetic model does not open");

    memset(&tally, 0, sizeof(tally));
    for (c = 0; c < (uint32_t)CASES; c++) {
        const Case *t = &cases[c];
        WexNr2Layout layout;
        unsigned char *scales[3], *weights[3];
        const unsigned char *cscales[3], *cweights[3];
        unsigned char anchor[3];
        uint32_t flags = t->cz ? WEX_NR3_FLAG_CZ : 0u, owner = (c + 1u) << 16 | (c * 7u + 3u);
        size_t capacity = 4096u, written = 0, total;
        for (m = 0; m < 3u; m++) {
            uint32_t blocks = t->rows[m], b, k;
            layout.scale_count[m] = blocks; layout.symbol_count[m] = blocks * 32u;
            scales[m] = (unsigned char *)xmalloc(blocks); weights[m] = (unsigned char *)xmalloc((size_t)blocks * 16u);
            for (b = 0; b < blocks; b++) {
                /* 0, 1: any magnitude; 2: an even block; 3: small magnitudes, 0..4 in ceil-form data and 0..5
                 * elsewhere (a plain matrix may hold a block of maximum 5, which must stay as it is) */
                uint32_t kind = rng() % 4u;
                scales[m][b] = (unsigned char)(118u + rng() % 7u);
                for (k = 0; k < 32u; k++) {
                    uint32_t a = kind == 3u ? rng() % (t->ceil ? 5u : 6u) : rng() & 7u, sign = rng() & 8u;
                    if (kind == 2u) a = k == 5u ? 7u : wex_nr3_double[rng() % 6u];
                    if (t->ceil && kind < 2u && a == 5u) a = 6u;            /* the gate: no block with max 5 */
                    if (t->ceil && kind < 2u && k == 0u) a = 7u;
                    if (t->cz && !a) sign = 0;                               /* no -0 */
                    weights[m][(size_t)b * 16u + k / 2u] |= (unsigned char)((a | sign) << ((k & 1u) * 4u));
                }
            }
            cscales[m] = scales[m]; cweights[m] = weights[m];
            capacity += (size_t)blocks * 96u + 64u;   /* above 14 bits per symbol and 6 bytes per stream */
        }
        if (t->ceil) {
            uint32_t stats[3][3];
            flags |= wex_nr3_ceil_forward(&layout, scales, weights, stats);
            if ((flags & WEX_NR3_FLAG_CEIL_MASK) != WEX_NR3_FLAG_CEIL_MASK || !stats[0][0])
                DIE("selftest: case %u is not in ceil form", (unsigned)c);
        }
        if (t->broken == B_CEIL) {   /* the first halved block of matrix 1 gets coded scale 0 */
            uint32_t b;
            for (b = 0; b < layout.scale_count[1] && wex_nr3_block_max(weights[1] + (size_t)b * 16u, NULL) != 5u; b++) {}
            if (b == layout.scale_count[1]) DIE("selftest: no max-5 block");
            scales[1][b] = 0;
        }
        if (t->broken == B_CZ) weights[2][100] = (unsigned char)((weights[2][100] & 0x0fu) | 0x80u);
        for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[m], layout.scale_count[m]);
        records[c] = (unsigned char *)xmalloc(capacity);
        if (wex_nrx_encode(&model, &layout, cscales, cweights, t->weight_chunk, t->scale_chunk, anchor, WEX_NR3_MAGIC,
                           flags, owner, records[c], capacity - 4096u, &written)) DIE("selftest: case %u does not encode", (unsigned)c);
        if (t->broken == B_BYTE) records[c][written - 9u] ^= 0x40u;
        if (t->broken == B_OWNER) records[c][28] ^= 1u;
        if (t->broken >= B_TAIL)
            break_scale_stream(records[c], &written, &model, &layout, t->weight_chunk, t->scale_chunk, t->broken);
        total = (size_t)wex4_align(written);
        memset(&jobs[c], 0, sizeof(jobs[c]));
        jobs[c].layout = layout;
        if (wex_nrx_parse(records[c], total, &layout, model.id, WEX_NR3_MAGIC, &jobs[c].view))
            DIE("selftest: case %u does not parse", (unsigned)c);
        jobs[c].stored = records[c]; jobs[c].stored_bytes = (uint32_t)total; jobs[c].model = blob;
        for (m = 0; m < 3u; m++) { free(scales[m]); free(weights[m]); }
    }
    if (only >= (int)CASES) DIE("selftest: the cases are 0 to %d", (int)CASES - 1);
    if (only >= 0) { from = (uint32_t)only; to = from + 1u; }
    failures += run_batch(g, jobs + from, to - from, "selftest", &tally);
    for (c = from; c < to; c++) {
        const Case *t = &cases[c];
        /* the error word each case must end with. A changed byte ends as a wrong hash, as a stream that does
         * not end at its length, or as both in different streams: bit 1, bit 2 or both. The four cases
         * after it are built so that exactly one check can refuse them. */
        static const uint32_t expected[] = {0, ERR_CEIL, ERR_CZ, 0, ERR_HASH, ERR_STRUCT, ERR_STRUCT, ERR_STRUCT};
        int ok = t->broken == B_BYTE ? jobs[c].error && !(jobs[c].error & ~(ERR_HASH | ERR_STRUCT))
                                     : jobs[c].error == expected[t->broken];
        printf("selftest %u: %s: %u + %u streams, flags 0x%02x, error word 0x%x%s\n", (unsigned)c, t->name,
               (unsigned)jobs[c].view.scale_streams, (unsigned)jobs[c].view.weight_streams, (unsigned)jobs[c].view.flags,
               (unsigned)jobs[c].error, ok ? "" : "  <- not the expected one");
        if (!ok) failures++;
    }
    for (c = 0; c < (uint32_t)CASES; c++) free(records[c]);
    printf("selftest: %lu accepted with the reference's bytes (%lu blocks undone by the ceil-form inverse), %lu refused by both, "
           "%lu differences\n", tally.accepted, tally.ceil_blocks, tally.refused, tally.different);
    if (only < 0 && (tally.accepted != (unsigned long)VALID || tally.refused != (unsigned long)BROKEN || !tally.ceil_blocks)) {
        printf("selftest: expected %d accepted, %d refused\n", (int)VALID, (int)BROKEN); failures++;
    }
    return failures;
}

int main(int argc, char **argv) {
    Gpu gpu;
    unsigned long failures = 0;
    int i, mode = -1, first = 2;
    uint32_t copies = 0;
    if (argc >= 2 && !strcmp(argv[1], "selftest") && argc <= 3) mode = 3;
    else if (argc >= 3 && !strcmp(argv[1], "decode")) mode = 0;
    else if (argc >= 3 && !strcmp(argv[1], "reject")) mode = 1;
    else if (argc >= 5 && !strcmp(argv[1], "mutate")) {
        mode = 2; first = 4;
        copies = (uint32_t)strtoul(argv[2], NULL, 10);
        rng_state = (uint64_t)strtoul(argv[3], NULL, 10) * UINT64_C(0x9e3779b97f4a7c15) + 1u;
        if (!copies) mode = -1;
    }
    else if ((argc == 4 || argc == 5) && !strcmp(argv[1], "bench")) {
        mode = 4;
        copies = (uint32_t)strtoul(argv[2], NULL, 10);
        if (!copies) mode = -1;
    }
    if (mode < 0) {
        fprintf(stderr, "usage: wnr3_vk selftest [CASE] | decode FILE.wex4 ... | reject FILE.wex4 ... | mutate N SEED FILE.wex4 ... |\n"
                        "       bench RUNS FILE.wex4 [RECORDS]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);   /* a log that is a pipe shows how far a run came */
    gpu_open(&gpu);
    if (mode == 3) failures += selftest(&gpu, argc == 3 ? atoi(argv[2]) : -1);
    else if (mode == 4) failures += bench(&gpu, argv[3], copies, argc == 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : MAX_BATCH);
    else for (i = first; i < argc; i++) failures += run_file(&gpu, argv[i], mode, copies);
    gpu_close(&gpu);
    if (gpu.validate) {
        printf("validation layer: %lu messages\n", validation_messages);
        failures += validation_messages;
    }
    printf(failures ? "wnr3_vk: FAILED (%lu)\n" : "wnr3_vk: ok\n", failures);
    return failures ? 1 : 0;
}
