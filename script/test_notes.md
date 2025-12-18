| Test Name     | Problem              | Notes                |
|---------------|----------------------|----------------------|
|7299310        |structure only, structural loss is normalized | outcome is pretty standard and can act as baseline |
|7318368        |baseline joint clip + structure | ended up just looking like structure loss in isolation? |
|7319853        |restructured to match ous and manideep: clip is scaled by current compliance (weighted by alpha) | I can tell that there's both structure and clip it's just poorly resolved. Interesting|
|7322737        |implemented some changes to file saving conventions + viewing progress | worked |
|7323156        |allowed upsampling after first iteration, implemented some speed optimizations (FP32 tensors across models + return from numpy bridge for gradients), optimizations allowing pre-blur for conv etc. optimization follows inverse scaling (structure strong at first then clip increases) | it's giving butterfly for sure but we killed upsampling |
|7323826        |tried to fix upsampling (previous code didn't upsample! compiling the CNN froze the sizes)| upsampling works and its beautiful |
|7324123        |replaced original backend with pyFANTOM| | works very nicely similar quality as above * CORRECTION WAS USING LEGACY |
|7324279        |second test with pyFANTOM (git add) | works very nicely different result thanks to cnn * CORRECTION WAS USING LEGACY |
|7326076        |implemented filter schedule + beta schedule for different resolutions using legacy| SADLY DID NOT OUTPUT A RESULT but I saw a clip and there was a butterfly floating in empty space |
|7326473        |actually using pyFANTOM for this one |
|7326534        |forcing a legacy run to see effect of different resolutions|