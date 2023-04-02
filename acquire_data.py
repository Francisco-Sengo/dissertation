from pathlib import PureWindowsPath, Path
import starfinder as sf
import glob

# Path to the original .fits files and conversion to the user OS
fits_files = PureWindowsPath("./stars/fits/*.fits")
fits_files = Path(fits_files)
fits_files = glob.glob(str(fits_files))

sf.div_files(fits_files)


