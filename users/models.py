from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class CustomUser(AbstractUser):
    USER_TYPE_PLAYER = 'player'
    USER_TYPE_TEAM = 'team'

    USER_TYPE_CHOICES = (
        (USER_TYPE_PLAYER, 'Player'),
        (USER_TYPE_TEAM, 'Team'),
    )

    email = models.EmailField(unique=True)
    user_type = models.CharField(
        max_length=10,
        choices=USER_TYPE_CHOICES
    )
    faceit_nickname = models.CharField(max_length=50, blank=True, null=True)  # теперь может быть пустым для команд
    registration_date = models.DateTimeField(default=timezone.now)

    REQUIRED_FIELDS = ['email', 'user_type']

    def __str__(self):
        return self.username

    def save(self, *args, **kwargs):
        # Для игроков проверяем Faceit никнейм
        if self.user_type == self.USER_TYPE_PLAYER and self.faceit_nickname:
            from .faceit import faceit_api
            player_data = faceit_api.get_player_by_nickname(self.faceit_nickname)
            if player_data and 'player_id' in player_data:
                # Здесь можно сохранить player_id если нужно
                pass
        super().save(*args, **kwargs)


class PlayerProfile(models.Model):
    # Позиции на картах
    MAP_CHOICES = [
        ('mirage', 'Mirage'),
        ('dust2', 'Dust 2'),
        ('anubis', 'Anubis'),
        ('ancient', 'Ancient'),
        ('nuke', 'Nuke'),
        ('inferno', 'Inferno'),
        ('overpass', 'Overpass'),
    ]

    # CT позиции для каждой карты
    CT_POSITION_CHOICES = {
        'mirage': [
            ('mirage_ct_b_anchor', 'Опорник Б'),
            ('mirage_ct_short', 'Шорт'),
            ('mirage_ct_window', 'Окно'),
            ('mirage_ct_con', 'Кон'),
            ('mirage_ct_a_anchor', 'Опорник А'),
        ],
        'dust2': [
            ('dust2_ct_long', 'Лонг'),
            ('dust2_ct_mid', 'Мид'),
            ('dust2_ct_short', 'Шорт'),
            ('dust2_ct_b', 'Б'),
            ('dust2_ct_rotate', 'Ротейт'),
        ],
        'anubis': [
            ('anubis_ct_b_anchor', 'Опорник Б'),
            ('anubis_ct_con', 'Кон'),
            ('anubis_ct_mid', 'Мид'),
            ('anubis_ct_a_anchor', 'Опорник А'),
            ('anubis_ct_rotate', 'Ротейт'),
        ],
        'nuke': [
            ('nuke_ct_outside', 'Улица'),
            ('nuke_ct_main', 'Мейн'),
            ('nuke_ct_a_anchor', 'Опорник А'),
            ('nuke_ct_rotate', 'Ротейт'),
            ('nuke_ct_ramp', 'Рамп'),
        ],
        'ancient': [
            ('ancient_ct_b_anchor', 'Опорник Б'),
            ('ancient_ct_cave', 'Кейв'),
            ('ancient_ct_mid', 'Мид'),
            ('ancient_ct_donate', 'Донат'),
            ('ancient_ct_a_anchor', 'Опорник А'),
        ],
        'inferno': [
            ('inferno_ct_b_anchor', 'Опорник Б'),
            ('inferno_ct_rotate', 'Ротейт'),
            ('inferno_ct_long', 'Лонг'),
            ('inferno_ct_short', 'Шорт'),
            ('inferno_ct_aps', 'АПС'),
        ],
        'overpass': [
            ('overpass_ct_b_anchor', 'Опорник Б'),
            ('overpass_ct_rotate', 'Ротейт'),
            ('overpass_ct_mid', 'Мид'),
            ('overpass_ct_con', 'Кон'),
        ],
    }

    # T роли
    T_ROLE_CHOICES = [
        ('lurker', 'Люркер'),
        ('entry', 'Ентри'),
        ('support', 'Сапорт'),
    ]

    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='player_profile')

    # Faceit данные
    faceit_player_id = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=10, blank=True)
    level = models.IntegerField(null=True, blank=True)
    elo = models.IntegerField(null=True, blank=True)
    skill_level = models.IntegerField(null=True, blank=True)
    faceit_url = models.URLField(blank=True)
    avatar = models.URLField(blank=True)

    # Статистика
    matches = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)
    average_kd = models.FloatField(default=0.0)
    average_hs = models.FloatField(default=0.0)
    current_win_streak = models.IntegerField(default=0)
    longest_win_streak = models.IntegerField(default=0)

    # Игровые характеристики
    is_igl = models.BooleanField(default=False)
    can_awp = models.BooleanField(default=False)

    # Позиции на картах (CT)
    mirage_ct_position = models.CharField(max_length=50, blank=True)
    dust2_ct_position = models.CharField(max_length=50, blank=True)
    anubis_ct_position = models.CharField(max_length=50, blank=True)
    nuke_ct_position = models.CharField(max_length=50, blank=True)
    ancient_ct_position = models.CharField(max_length=50, blank=True)
    inferno_ct_position = models.CharField(max_length=50, blank=True)
    overpass_ct_position = models.CharField(max_length=50, blank=True)

    # Роль на T стороне
    t_role = models.CharField(max_length=20, choices=T_ROLE_CHOICES, blank=True)

    # Дополнительная информация
    description = models.TextField(blank=True)
    steam_id = models.CharField(max_length=50, blank=True)
    last_faceit_update = models.DateTimeField(null=True, blank=True)
    profile_created = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'PlayerProfile: {self.user.username}'

    @staticmethod
    def _split_positions(value):
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [item for item in value.split(',') if item]

    def get_ct_position_display(self, map_name):
        """Получить отображаемое название позиции для карты"""
        positions = self.CT_POSITION_CHOICES.get(map_name, [])
        position_lookup = dict(positions)
        stored_value = getattr(self, f'{map_name}_ct_position', '')
        selected_positions = self._split_positions(stored_value)
        labels = [position_lookup.get(position, position) for position in selected_positions]
        return ', '.join(labels) if labels else 'Не указано'

    def get_mirage_ct_position_display(self):
        return self.get_ct_position_display('mirage')

    def get_dust2_ct_position_display(self):
        return self.get_ct_position_display('dust2')

    def get_anubis_ct_position_display(self):
        return self.get_ct_position_display('anubis')

    def get_nuke_ct_position_display(self):
        return self.get_ct_position_display('nuke')

    def get_ancient_ct_position_display(self):
        return self.get_ct_position_display('ancient')

    def get_inferno_ct_position_display(self):
        return self.get_ct_position_display('inferno')

    def get_overpass_ct_position_display(self):
        return self.get_ct_position_display('overpass')

    def get_t_role_display(self):
        """Получить отображаемое название T роли"""
        for role_code, role_name in self.T_ROLE_CHOICES:
            if role_code == self.t_role:
                return role_name
        return 'Не указано'


class TeamProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='team_profile')

    # Основная информация
    team_name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    # Требования к игрокам
    looking_for_igl = models.BooleanField(default=False)
    looking_for_awper = models.BooleanField(default=False)

    # Поиск по позициям
    needed_positions = models.TextField(blank=True)  # JSON или текст с позициями

    # Настройки
    is_active = models.BooleanField(default=True)
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'TeamProfile: {self.team_name}'
