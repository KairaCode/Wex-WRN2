/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* A host of its own for crash_min.comp (README, "The llvmpipe crash"; docs/WNR3-VULKAN.md, section 6): no
 * record, no model and no WNR3 code, a few small storage buffers and one dispatch.
 *
 *   crash_min SHADER.spv [COUNT [GROUPS]]
 *
 * The buffers the shader reads and writes:
 *   binding 1  16,384 words of zeros, except the 257 16-bit values the shader's search reads (u16 number
 *              16 + 3 * 567 * 17 + s holds s * 64, s = 0..256: an even CDF, as a scale CDF of a model blob
 *              would be at that place)
 *   binding 2  32 words of zeros, except word 12 = COUNT (default 256, the scale chunk size that the selftest
 *              records of wnr3_vk.c hold there)
 *   binding 5  one word, zero
 * then vkCmdDispatch(GROUPS, 1, 1) (default 1 workgroup of 32 invocations), one submit and a wait. Every read
 * of the shader is in bounds. The rest of the set-up is that of wnr3_vk.c unless said otherwise:
 *   CRASH_MIN_BINDINGS=012345 (default)  the bindings of the descriptor set layout, digits 0 to 5 that
 *                                   include 1, 2 and 5: by default 0 to 5, as in wnr3_vk.c. A binding that
 *                                   crash_min.comp does not declare (0, 3, 4) gets a buffer of 1,024 zero words
 *   CRASH_MIN_PUSH=1 (default)      the pipeline layout has a push constant range of 4 bytes, as in wnr3_vk.c,
 *                                   and the command buffer pushes 0 (crash_min.comp reads none). 0: no range
 * WNR3_VK_DEVICE (an index or a part of the device name), WNR3_VK_VALIDATE=1 (the Khronos validation layer,
 * any message is a failure) and WNR3_VK_ROBUST=1 (robustBufferAccess) as in wnr3_vk.c. Prints the device and
 * the set-up it uses and, when the dispatch came back, the word of binding 5.
 * Exit 0 when it came back, 1 on a message of the validation layer, 2 on a setup error. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define DIE(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
#define VK(call) do { VkResult vk_r = (call); if (vk_r != VK_SUCCESS) DIE("%s: VkResult %d", #call, (int)vk_r); } while (0)

#define TAB_WORDS 16384u
#define PLAN_WORDS 32u
#define OTHER_WORDS 1024u
#define CDF_AT (16u + 3u * 567u * 17u)   /* u16 number of value 0 of the CDF the shader searches */

typedef struct { VkBuffer buffer; VkDeviceMemory memory; unsigned char *map; } Buffer;

static unsigned long messages;

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)severity; (void)types; (void)user;
    messages++;
    fprintf(stderr, "validation: %s\n", data->pMessage);
    return VK_FALSE;
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

/* a host-visible, coherent storage buffer of bytes zero bytes, mapped */
static void buffer_create(VkDevice device, const VkPhysicalDeviceMemoryProperties *memory, VkDeviceSize bytes, Buffer *b) {
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBufferCreateInfo info;
    VkMemoryRequirements need;
    VkMemoryAllocateInfo alloc;
    void *map;
    uint32_t i;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(device, &info, NULL, &b->buffer));
    vkGetBufferMemoryRequirements(device, b->buffer, &need);
    for (i = 0; i < memory->memoryTypeCount; i++)
        if ((need.memoryTypeBits >> i & 1u) && (memory->memoryTypes[i].propertyFlags & want) == want) break;
    if (i == memory->memoryTypeCount) DIE("no host-visible memory type");
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = i;
    VK(vkAllocateMemory(device, &alloc, NULL, &b->memory));
    VK(vkBindBufferMemory(device, b->buffer, b->memory, 0));
    VK(vkMapMemory(device, b->memory, 0, VK_WHOLE_SIZE, 0, &map));
    b->map = (unsigned char *)map;
    memset(b->map, 0, (size_t)bytes);
}

static void put_u16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

int main(int argc, char **argv) {
    static const char *const layer = "VK_LAYER_KHRONOS_validation";
    static const char *const debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    const char *choice = getenv("WNR3_VK_DEVICE"), *validate_text = getenv("WNR3_VK_VALIDATE"),
               *robust = getenv("WNR3_VK_ROBUST"), *bindings_text = getenv("CRASH_MIN_BINDINGS"),
               *push_text = getenv("CRASH_MIN_PUSH");
    int validate = validate_text && validate_text[0] == '1', push = !push_text || push_text[0] != '0';
    uint32_t count = argc >= 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 256u;
    uint32_t groups = argc >= 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 1u;
    uint32_t devices_count = 16u, families_count = 32u, i, index = 0, family, *code, word, zero = 0, used[6], n = 0;
    VkApplicationInfo app;
    VkInstanceCreateInfo instance_info;
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice devices[16];
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceFeatures features, wanted;
    VkQueueFamilyProperties families[32];
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkDevice device;
    VkQueue queue;
    VkDescriptorSetLayoutBinding bindings[6];
    VkDescriptorSetLayoutCreateInfo set_layout_info;
    VkDescriptorSetLayout set_layout;
    VkPushConstantRange range;
    VkPipelineLayoutCreateInfo layout_info;
    VkPipelineLayout layout;
    VkShaderModuleCreateInfo module_info;
    VkShaderModule module;
    VkComputePipelineCreateInfo pipeline_info;
    VkPipeline pipeline;
    VkDescriptorPoolSize pool_size;
    VkDescriptorPoolCreateInfo pool_info;
    VkDescriptorPool pool;
    VkDescriptorSetAllocateInfo set_info;
    VkDescriptorSet set;
    VkDescriptorBufferInfo buffer_info[6];
    VkWriteDescriptorSet writes[6];
    VkCommandPoolCreateInfo command_pool_info;
    VkCommandPool command_pool;
    VkCommandBufferAllocateInfo cmd_info;
    VkCommandBuffer cmd;
    VkCommandBufferBeginInfo begin;
    VkFenceCreateInfo fence_info;
    VkFence fence;
    VkSubmitInfo submit;
    Buffer buffers[6];   /* by binding */
    FILE *f;
    long size;
    float priority = 1.0f;

    if (argc < 2 || argc > 4 || !groups) {
        fprintf(stderr, "usage: crash_min SHADER.spv [COUNT [GROUPS]]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);   /* the lines before a crash reach the log */
    f = fopen(argv[1], "rb");
    if (!f) DIE("%s: cannot open", argv[1]);
    if (fseek(f, 0, SEEK_END) || (size = ftell(f)) <= 0 || (size & 3) || fseek(f, 0, SEEK_SET)) DIE("%s: not SPIR-V", argv[1]);
    code = (uint32_t *)malloc((size_t)size);
    if (!code || fread(code, 1u, (size_t)size, f) != (size_t)size) DIE("%s: short read", argv[1]);
    fclose(f);

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "crash_min";
    app.apiVersion = VK_API_VERSION_1_0;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    if (validate) {
        instance_info.enabledLayerCount = 1u; instance_info.ppEnabledLayerNames = &layer;
        instance_info.enabledExtensionCount = 1u; instance_info.ppEnabledExtensionNames = &debug_extension;
    }
    VK(vkCreateInstance(&instance_info, NULL, &instance));
    if (validate) {
        VkDebugUtilsMessengerCreateInfoEXT info;
        PFN_vkCreateDebugUtilsMessengerEXT create = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (!create) DIE("the validation layer is on but VK_EXT_debug_utils is missing");
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = on_message;
        VK(create(instance, &info, NULL, &messenger));
    }
    {
        VkResult r = vkEnumeratePhysicalDevices(instance, &devices_count, devices);
        if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !devices_count) DIE("no Vulkan device");
    }
    if (choice && choice[0] && !choice[strspn(choice, "0123456789")]) {
        index = (uint32_t)strtoul(choice, NULL, 10);
    } else if (choice && choice[0]) {
        for (index = 0; index < devices_count; index++) {
            vkGetPhysicalDeviceProperties(devices[index], &properties);
            if (name_has(properties.deviceName, choice)) break;
        }
    }
    if (index >= devices_count) DIE("WNR3_VK_DEVICE=%s: no such device", choice);
    vkGetPhysicalDeviceProperties(devices[index], &properties);
    printf("device %u: %s (Vulkan %u.%u.%u, vendor 0x%04x, driver version 0x%08x)\n", (unsigned)index, properties.deviceName,
           (unsigned)VK_VERSION_MAJOR(properties.apiVersion), (unsigned)VK_VERSION_MINOR(properties.apiVersion),
           (unsigned)VK_VERSION_PATCH(properties.apiVersion), (unsigned)properties.vendorID, (unsigned)properties.driverVersion);
    vkGetPhysicalDeviceMemoryProperties(devices[index], &memory);
    vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families_count, families);
    for (family = 0; family < families_count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT); family++) {}
    if (family == families_count) DIE("the device has no compute queue");

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1u;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1u;
    device_info.pQueueCreateInfos = &queue_info;
    if (robust && robust[0] == '1') {
        vkGetPhysicalDeviceFeatures(devices[index], &features);
        if (!features.robustBufferAccess) DIE("WNR3_VK_ROBUST=1 but the device has no robustBufferAccess");
        memset(&wanted, 0, sizeof(wanted));
        wanted.robustBufferAccess = VK_TRUE;
        device_info.pEnabledFeatures = &wanted;
        printf("robustBufferAccess is on\n");
    }
    VK(vkCreateDevice(devices[index], &device_info, NULL, &device));
    vkGetDeviceQueue(device, family, 0, &queue);

    /* the bindings of the set layout, in increasing order */
    if (!bindings_text || !bindings_text[0]) bindings_text = "012345";
    if (bindings_text[strspn(bindings_text, "012345")]) DIE("CRASH_MIN_BINDINGS=%s: digits 0 to 5 only", bindings_text);
    for (i = 0; i < 6u; i++) if (strchr(bindings_text, (int)('0' + i))) used[n++] = i;
    if (!strchr(bindings_text, '1') || !strchr(bindings_text, '2') || !strchr(bindings_text, '5'))
        DIE("CRASH_MIN_BINDINGS=%s: bindings 1, 2 and 5 are needed", bindings_text);
    for (i = 0; i < n; i++) {
        uint32_t b = used[i];
        buffer_create(device, &memory, (b == 1u ? TAB_WORDS : b == 2u ? PLAN_WORDS : b == 5u ? 1u : OTHER_WORDS) * 4u, &buffers[b]);
    }
    for (i = 0; i <= 256u; i++) put_u16(buffers[1].map + (CDF_AT + i) * 2u, i * 64u);
    buffers[2].map[48] = (unsigned char)count; buffers[2].map[49] = (unsigned char)(count >> 8);
    buffers[2].map[50] = (unsigned char)(count >> 16); buffers[2].map[51] = (unsigned char)(count >> 24);
    printf("set-up: bindings");
    for (i = 0; i < n; i++) printf(" %u", (unsigned)used[i]);
    printf(" in the set layout; %s\n", push ? "a push constant range of 4 bytes, 0 pushed" : "no push constant range");

    memset(bindings, 0, sizeof(bindings));
    for (i = 0; i < n; i++) {
        bindings[i].binding = used[i];
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1u;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    memset(&set_layout_info, 0, sizeof(set_layout_info));
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = n;
    set_layout_info.pBindings = bindings;
    VK(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL, &set_layout));
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; range.offset = 0; range.size = 4u;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1u;
    layout_info.pSetLayouts = &set_layout;
    if (push) { layout_info.pushConstantRangeCount = 1u; layout_info.pPushConstantRanges = &range; }
    VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = (size_t)size;
    module_info.pCode = code;
    VK(vkCreateShaderModule(device, &module_info, NULL, &module));
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1u, &pipeline_info, NULL, &pipeline));

    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = n;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1u; pool_info.poolSizeCount = 1u; pool_info.pPoolSizes = &pool_size;
    VK(vkCreateDescriptorPool(device, &pool_info, NULL, &pool));
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1u;
    set_info.pSetLayouts = &set_layout;
    VK(vkAllocateDescriptorSets(device, &set_info, &set));
    memset(writes, 0, sizeof(writes));
    for (i = 0; i < n; i++) {
        buffer_info[i].buffer = buffers[used[i]].buffer; buffer_info[i].offset = 0; buffer_info[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = used[i]; writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_info[i];
    }
    vkUpdateDescriptorSets(device, n, writes, 0, NULL);

    memset(&command_pool_info, 0, sizeof(command_pool_info));
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = family;
    VK(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool));
    memset(&cmd_info, 0, sizeof(cmd_info));
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1u;
    VK(vkAllocateCommandBuffers(device, &cmd_info, &cmd));
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1u, &set, 0, NULL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (push) vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4u, &zero);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    {   /* the shader's write before the host reads the word */
        VkMemoryBarrier b;
        memset(&b, 0, sizeof(b));
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1u, &b, 0, NULL, 0, NULL);
    }
    VK(vkEndCommandBuffer(cmd));
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK(vkCreateFence(device, &fence_info, NULL, &fence));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers = &cmd;
    printf("dispatch: %u workgroup%s, loop count %u\n", (unsigned)groups, groups == 1u ? "" : "s", (unsigned)count);
    VK(vkQueueSubmit(queue, 1u, &submit, fence));
    VK(vkWaitForFences(device, 1u, &fence, VK_TRUE, UINT64_MAX));
    word = (uint32_t)buffers[5].map[0] | (uint32_t)buffers[5].map[1] << 8 | (uint32_t)buffers[5].map[2] << 16 |
           (uint32_t)buffers[5].map[3] << 24;

    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    for (i = 0; i < n; i++) {
        vkDestroyBuffer(device, buffers[used[i]].buffer, NULL);
        vkFreeMemory(device, buffers[used[i]].memory, NULL);
    }
    vkDestroyDevice(device, NULL);
    if (messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(instance, messenger, NULL);
    }
    vkDestroyInstance(instance, NULL);
    free(code);
    if (validate) printf("validation layer: %lu messages\n", messages);
    printf("crash_min: the dispatch came back, word of binding 5 0x%x\n", (unsigned)word);
    return messages ? 1 : 0;
}
