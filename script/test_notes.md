| Test Name     | Problem              | Notes                |
|---------------|----------------------|----------------------|
|7299310        |structure only, structural loss is normalized | outcome is pretty standard and can act as baseline |
|7318368        |baseline joint clip + structure | ended up just looking like structure loss in isolation? |
|7319853        |restructured to match ous and manideep: clip is scaled by current compliance (weighted by alpha) | I can tell that there's both structure and clip it's just poorly resolved. Interesting|
|7322737        |implemented some changes to file saving conventions + viewing progress | worked |
|7323156        |allowed upsampling after first iteration, implemented some speed optimizations (FP32 tensors across models + return from numpy bridge for gradients), optimizations allowing pre-blur for conv etc. optimization follows inverse scaling (structure strong at first then clip increases) | it's giving butterfly for sure but we killed upsampling |
|7323826        |tried to fix upsampling (previous code didn't upsample! compiling the CNN froze the sizes)| upsampling works and its beautiful |
|7324123        |replaced original backend with pyFANTOM|