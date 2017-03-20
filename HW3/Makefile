.PHONY: all show

all: download prepare split visualize show

download :
	curl https://transfer.sh/hl42s/shortscans.npy -o scans.npy

prepare : scans.npy
	/usr/bin/env python prepare.py

split : data.txt
	/usr/bin/env python split.py

visualize : train.txt scans.npy
	/usr/bin/env python visualize.py

show : visualization.png
	@if [[ -x `which open` ]]; then \
		open visualization.png; \
	elif [[ -x `which xdg-open` ]]; then \
		xdg-open visualization.png; \
	fi

clean: 
	rm -f *.pyc
	rm -f visualization.png
	rm -f data.txt train.txt test.txt

mrproper: clean
	rm -f scans.npy

