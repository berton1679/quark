#!/bin/bash
files=`find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -not -path "./build/*"`
clang-format -style=file -ni ${files}
clang-format -style=file -i ${files}
