# Docx2Nite
This script can be used to convert bulleted lists written in DOCX to the NITE XML format.

## Building
```
./gradlew shadowJar
```
The JAR will be located at ```build/libs/docx2nite-all.jar```

## Usage
```
./docx2nite-all.jar DOCX_FILE_PATH
```
The generated XML file will be saved with suffix ".annotation.xml".

## Examples
See ```examples/```.
