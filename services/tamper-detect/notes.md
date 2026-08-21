# Day 1 Understanding Checkpoints

- **Why does a black image have low Laplacian variance?**
  A completely black image has uniform pixel values (zero intensity change). Because there are no edges or gradients, the variance of the second-order derivative is essentially 0.

- **Why do we require the histogram entropy of a textured image to be > 4 bits, not > 7?**
  Real-world scenes contain smooth gradients, shadows, and uniform areas that lower overall entropy. A threshold of > 4 bits filters out flat/blocked frames while letting genuine textured photos pass (whereas > 7 bits would wrongly reject valid real-world images).