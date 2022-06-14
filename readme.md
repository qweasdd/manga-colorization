UPD. See the [improved version](https://github.com/qweasdd/manga-colorization-v2).

# Automatic colorization

1. Download [generator](https://drive.google.com/file/d/1Oo6ycphJ3sUOpDCDoG29NA5pbhQVCevY/view?usp=sharing),  [extractor](https://drive.google.com/file/d/12cbNyJcCa1zI2EBz6nea3BXl21Fm73Bt/view?usp=sharing) and [denoiser ](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put generator and extractor weights in `model` and denoiser weights in `denoising/models`.
2. To colorize image, folder of images, `.cbz` or `.cbr` file, use the following command:
```
$ python inference.py -p "path to file or folder"
```

# Manual colorization with color hints

1. Download [colorizer](https://drive.google.com/file/d/1BERrMl9e7cKsk9m2L0q1yO4k7blNhEWC/view?usp=sharing) and [denoiser ](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put colorizer weights in `model` and denoiser weights in `denoising/models`.
2.  Run gunicorn server with:
```
$ ./run_drawing.sh
```
3. Open `localhost:5000` with a browser.

# References
1. Extractor weights are taken from https://github.com/blandocs/Tag2Pix/releases/download/release/model.pth
2. Denoiser weights are taken from http://www.ipol.im/pub/art/2019/231.
