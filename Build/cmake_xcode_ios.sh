#!/bin/sh

cmake -S .. -B XCode_iOS -DCMAKE_SYSTEM_NAME=iOS -GXcode
open XCode_iOS/WulfNetEngine.xcodeproj
