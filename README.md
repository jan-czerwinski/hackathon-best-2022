# HACKATRON 2022
## ZADANIE KWALIFIKACYJNE

Należy napisać opracować własną implementację filtra wykrywającego krawędzie (https://en.wikipedia.org/wiki/Edge_detection) na obrazach barwnych w dowolnie wybranym przez siebie środowisku. Oceniana będzie optymalność oraz przejrzystość kodu, a także prędkość wykonania algorytmu. Wykorzystywanie gotowych rozwiązań, t.j. gotowych bibliotek realizujących w/w algorytm, lub gotowych funkcji znacznie ułatwiających pracę algorytmu, będzie skutkowało w obniżeniu punktacji. Plagiaty lub widocznie podobne kody u różnych drużyn również będą skutkowały obniżeniem punktacji.

## How to setup the project
- Create virtual environment ```python3 -m venv env```
- Activate virtual environment ```source ./env/bin/activate```
- Install required modules ```pip install -r requirements.txt```
- Run script ```make run```

## Run with custom images
- Activate virtual environment ```source ./env/bin/activate```
- ```python ./src/main.py --image_path path_to_image.jpg --save_path output_path.jpg```
