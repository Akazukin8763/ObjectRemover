## Prerequisites

```
pip install -r requirement.txt

git clone https://github.com/sczhou/ProPainter.git
```

Note that we tested on Python 3.9 / Pytorch 1.12.1 / CUDA11.3

## Result

<table>
  <tr>
    <th>&nbsp;</th>
    <th>Frame 300</th>
    <th>Frame 390</th>
    <th>Frame 480</th>
  </tr>
  <tr>
    <td>Original</td>
    <td><img src="assets/zebra_frame_300.png"></td>
    <td><img src="assets/zebra_frame_390.png"></td>
    <td><img src="assets/zebra_frame_480.png"></td>
  </tr>
  <tr>
    <td>Learning based</td>
    <td><img src="assets/zebra_learning_based_frame_300.png"></td>
    <td><img src="assets/zebra_learning_based_frame_390.png"></td>
    <td><img src="assets/zebra_learning_based_frame_480.png"></td>
  </tr>
  <tr>
    <td>Non-learning based</td>
    <td><img src="assets/zebra_non_learning_based_frame_300.png"></td>
    <td><img src="assets/zebra_non_learning_based_frame_390.png"></td>
    <td><img src="assets/zebra_non_learning_based_frame_480.png"></td>
  </tr>
</table>
