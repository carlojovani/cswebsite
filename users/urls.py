from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from . import views

urlpatterns = [
    # Главная страница
    path('', views.home, name='home'),

    # Регистрация (многошаговая)
    path('register/', views.register, name='register'),
    path('register/step1/', views.register_step1, name='register_step1'),
    path('register/step2/', views.register_step2, name='register_step2'),
    path('register/step3/', views.register_step3, name='register_step3'),

    # Аутентификация
    path('login/', LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),

    # Профили
    path('profile/<int:user_id>/', views.profile, name='profile'),
    path('profile/update-faceit/', views.update_faceit_data, name='update_faceit_data'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),  # ДОБАВЬТЕ ЭТУ СТРОКУ

    # Поиск (с ограничениями по типу пользователя)
    path('search/player/', views.search_player, name='search_player'),
    path('search/team/', views.search_team, name='search_team'),

    # Детальная информация (с ограничениями)
    path('player/<int:player_id>/', views.player_details, name='player_details'),
    path('team/<int:team_id>/', views.team_details, name='team_details'),
]