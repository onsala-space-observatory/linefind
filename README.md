# linefind

Contains a class for doing line / continuum identification in image cubes.

It can be used as a command line tool but the intended usage is from within Python.
If you want to experiment with different cutoffs and padding widths it's much faster
to do this in python as the SNR calculations only have to be done once.

If the cube has been corrected for a primary beam and consequently has non-flat noise
it's better to undo this by providing the `beam='beam.fits'` option.

## Example
```
>>> from linefind import LineFinder
>>> lf = LineFinder('cube.fits', beam='beam.fits')
>>> lf.find_lines(sigma=5, pad=20) # First call will be slow
>>> lf.plot()
>>> lf.find_lines(sigma=6, pad=40) # Subsequent calls are instant
>>> lf.plot()
>>> lf.spw_string()
93~132;766~805;1206~1478;1506~1580;1865~1906;
```
