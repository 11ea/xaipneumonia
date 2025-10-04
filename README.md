**1- virtual env oluştur.**

python -m venv venv-adi

**2- aktive et.**

source venv-adi/Scripts/activate

**3- dependencyleri kur.**

sadece django var diye hatırlıyorum

**4- db oluştur.**

python manage.py makemigrations

_(güncellerken makemigrations yerine migrate)_

**5- admin panel için superuser oluştur.**

python manage.py createsuperuser

**6- sunucuları çalıştır.**

python manage.py runserver

_(ayrı bi terminalde)_

npm run dev

**7- model ekle.**

localhost/admin e giderek superuserle giriş yap,
model ekle.
