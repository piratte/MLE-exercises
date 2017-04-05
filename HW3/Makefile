.PHONY: all show

all: download prepare split visualize show

download :
	curl https://transfer.sh/fys5K/scans.npy.zip -o scans.npy.zip
	unzip scans.npy.zip
	rm -f scans.npy.zip

prepare : scans.npy
	/usr/bin/env python hists.py

split : hists.txt
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
	rm -f hists.txt header.txt train.txt test.txt

mrproper: clean
	rm -f scans.npy
