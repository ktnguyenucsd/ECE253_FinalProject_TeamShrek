**Wiener Filter**
***
To use wiener filter to deblur an image run the following commend
```bash
python ./wiener_filter.py INPUT_IMAGE DIRECTION(IN DEGREE) PIXEL_LENGTH LAMNDA
```
For example
```bash
python ./wiener_filter.py ./96.png 0 100 0.1
```
The result of image will be saved as INPUT_IMAGE_clean