#!/bin/bash
set -e
margs=4 ### number of mandatory args

print_usage() {
    echo "$0 is a utility script to generate colored tiles from raw data stored in GRIB files"
    echo "Copyright: Piero Ferrarese, piero@sciscry.ai"
    echo "Usage: $0 [options]"
    echo " "
    echo "Options: "
    echo "-h        show help"
    echo "-i        specify the input file to process, GRIB format"
    echo "-o        specify the output file to process, in GTiff format"
    echo "-b        specify the band to process, default 1"
    echo "-min      specify the minimum value of the data, to be used for normalisation. Default to stats retrieved from the input file with gdalinfo -stats"
    echo "-max      specify the maximum value of the data, to be used for normalisation. Default to stats retrieved from the input file with gdalinfo -stats"
}


if [ $# -lt $margs ]; then
    print_usage
    exit 1
fi
export BAND=1
export MIN_SCALE=
export MAX_SCALE= 
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    -i)
      shift
      if test $# -gt 0; then
        export IN_FILE=$1
      else
        echo "Please specify the input <file_name>.grib file"
        exit 1
      fi
      shift
      ;;
    -o)
      shift
      if test $# -gt 0; then
        export OUT_FILE=$1
      else
        echo "Please specify the output <file_name>.tif file"
        exit 1
      fi
      shift
      ;;
    -b)
      shift
      if test $# -gt 0; then
        export BAND=$1
      else
	  echo "Band not specified. Using default value of 1."
      fi
      shift
      ;;
    -min)
      shift
      if test $# -gt 0; then
        export MIN_SCALE=$1
      else
	  echo "Warning: not specifying min_scale. Using default."
      fi
      shift
      ;;
    -max)
      shift
      if test $# -gt 0; then
        export MAX_SCALE=$1
      else
	  echo "Warning: not specifying max_scale. Using default."	  
      fi
      shift
      ;;
    -date)
      shift
      if test $# -gt 0; then
        export DATE=$1
      else
        echo "Please specify the input <file_name>.grib file"
        exit 1
      fi
      shift
      ;;
    -hours)
      shift
      if test $# -gt 0; then
        export HOURS=$1
      else
        echo "Please specify the input <file_name>.grib file"
        exit 1
      fi
      shift
      ;;
    -name)
      shift
      if test $# -gt 0; then
        export NAME=$1
      else
        echo "Please specify the input <file_name>.grib file"
        exit 1
      fi
      shift
      ;;
    
    *)
	break
      ;;
  esac
done    

if [ -z $IN_FILE ]; then
    echo "Please specify the -i (input_file) option first"
    exit 1
fi

if [ -z $MIN_SCALE ]; then
    export MIN_SCALE=$(gdalinfo -stats $IN_FILE | grep Min= | awk '{print $1}' | cut -d= -f2-)
    echo $MIN_SCALE
    if [ -z $MIN_SCALE ]; then
	echo "Could not retrieve min_scale. Check your input file!"
    exit 1
    fi
fi

if [ -z $MAX_SCALE ]; then
    export MAX_SCALE=$(gdalinfo -stats $IN_FILE | grep Min= | awk '{print $2}' | cut -d= -f2-)
    if [ -z $MAX_SCALE ]; then
	echo "Could not retrieve max_scale. Check your input file!"
	exit 1
    fi
fi


gdal_translate -r bilinear -b $BAND -scale $MIN_SCALE $MAX_SCALE 0 255 -ot Byte -of GTiff $IN_FILE $OUT_FILE.tif
gdalbuildvrt -resolution highest -b 1 $OUT_FILE.vrt $OUT_FILE.tif
export VAR=$(awk '{printf "%s\\n", $0}' colortable.vrt)
sed -i -e "s|<VRTRasterBand.*|<VRTRasterBand dataType="Byte" band="1"> \\n $VAR|" $OUT_FILE.vrt
gdal_translate -b 1 -ot Byte  $OUT_FILE.vrt "${OUT_FILE}"_colored.mbtiles
echo "This is the name ${NAME}"
python3 extract_tiles.py -f $OUT_FILE -d $DATE -t $HOURS --name "${NAME}"
