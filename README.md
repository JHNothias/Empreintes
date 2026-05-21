# Empreintes
An algorithm to automatically find the most relevant audio descriptors to describe the structure of an audio recording, a corpus of recordings or an audio dataset, embedded in a metric space. WIP.

## Descriptors

This algorithm handles 3 types of descriptors:

- Primary audio descriptors : These are descriptors computed from the audio file directly, and are fine-grained.
- Secondary audio descriptors : These are derived from primary audio descriptors. For all primary descriptors, we can compute:
  - The variance of the descriptor within a running window
  - The approximate derivative of the descriptor (difference between neighboring values)
  - The cumulative sum <span style="color: yellow;">[Pending implementation]</span>
  - The novelty for a symmetric kernel, a homogenous to inhomogenous kernel and an inhomogenous to homogenous kernel, as proposed in [1]. <span style="color: yellow;">[Pending implementation]</span>
- Topological descriptors. <span style="color: yellow;">[Pending implementation]</span>



[1] F. Kaiser and G. Peeters, “Multiple hypotheses at multiple scales for audio novelty computation within music,” in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, Vancouver, BC, Canada: IEEE, May 2013, pp. 231–235. doi: 10.1109/ICASSP.2013.6637643.
