# Writes a copy of wex4/research/vulkan-decode/wnr3_vk.c (as of 2026-11-15) that says, for a ceil-form
# record whose decoded scale bytes differ from the reference, which of the differing bytes are decrements of
# the ceil-form inverse that were lost although a later block of the same 4-byte word was decremented too.
# Usage: python3 -I explain_diff.py SOURCE.c OUT.c
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
old = """                        tally->ceil_blocks += coded_scales[m][b] != scales[m][b];
            }"""
new = """                        tally->ceil_blocks += coded_scales[m][b] != scales[m][b];
                if (!same) {
                    unsigned long lost = 0, predicted = 0, other = 0;
                    for (m = 0; m < 3u; m++) {
                        const unsigned char *d = device + job->scale_at[m];
                        for (b = 0; b < job->layout.scale_count[m]; b++) {
                            uint32_t e, later = 0;
                            int dec = coded_scales[m][b] != scales[m][b];
                            for (e = b + 1u; e < ((b | 3u) + 1u) && e < job->layout.scale_count[m]; e++)
                                later |= coded_scales[m][e] != scales[m][e];
                            if (dec && later) predicted++;
                            if (d[b] != scales[m][b]) {
                                if (dec && d[b] == coded_scales[m][b] && later) lost++; else other++;
                            }
                        }
                    }
                    printf("  explained: %lu differing bytes are lost decrements with a later decrement in the same word; "
                           "%lu decrements of the record have one; %lu differing bytes are something else\\n", lost, predicted, other);
                }
            }"""
if s.count(old) != 1:
    sys.exit("explain_diff.py: the source has changed, the patch does not apply")
open(dst, "w", encoding="utf-8").write(s.replace(old, new))
