install:venv
	. venv/bin/activate; python3 -m install -upgrade pip;mkdir "Images";pip3 install -Ur requirements.txt
venv:
	test -d venv || python3 -m venv venv
Scenario1:
	python3 Scenario1.py
Scenario2:
	python3 Scenario2.py
Scenario3:
	python3 Scenario3.py
clean:
	rm -rf venv
	rm -rf Images
