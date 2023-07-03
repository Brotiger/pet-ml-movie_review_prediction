# Movie review prediction
Модель определяет по тексту отзыва на фильм положительный он или отрицательный
Датасет можно взять с сайта [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Используемые технологии
* Python - v3.10.6
* pip - v22.0.2

## Сборка и запуск
### Установка зависимостей
~~~
pip install -r requirements.txt
~~~

## Конфигурация
```bash
cp .env.example .env
```
_Прописать в `.env` параметры подключений._

```bash
source .env
```

### Запуск
~~~bash
python ./movie_review_prediction.py
~~~