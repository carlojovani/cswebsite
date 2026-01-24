from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from .models import CustomUser, PlayerProfile, TeamProfile
from .faceit import check_faceit_nickname


class RegistrationStep1Form(forms.Form):
    """Первая форма регистрации - выбор типа аккаунта"""
    USER_TYPE_CHOICES = [
        ('player', '👤 Игрок'),
        ('team', '👥 Команда'),
    ]

    user_type = forms.ChoiceField(
        choices=USER_TYPE_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Я хочу зарегистрироваться как:',
        required=True
    )


class PlayerRegistrationForm(UserCreationForm):
    """Форма регистрации игрока"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Введите ваш email'
        })
    )

    faceit_nickname = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ваш Faceit никнейм'
        })
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'faceit_nickname', 'password1', 'password2')

        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Имя пользователя на сайте'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Пароль'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Подтверждение пароля'})

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError('Этот email уже используется')
        return email

    def clean_faceit_nickname(self):
        faceit_nickname = self.cleaned_data.get('faceit_nickname')

        if CustomUser.objects.filter(faceit_nickname=faceit_nickname).exists():
            raise ValidationError('Этот Faceit никнейм уже зарегистрирован')

        if not check_faceit_nickname(faceit_nickname):
            raise ValidationError(
                'Faceit никнейм не найден. Убедитесь, что: '
                '1) Никнейм написан правильно '
                '2) У вас есть аккаунт на Faceit '
                '3) Вы играли в CS2 на Faceit'
            )

        return faceit_nickname

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.faceit_nickname = self.cleaned_data['faceit_nickname']
        user.user_type = CustomUser.USER_TYPE_PLAYER

        if commit:
            user.save()

        return user


class PlayerDetailsForm(forms.ModelForm):
    """Форма деталей игрока (второй шаг)"""
    # CT позиции для каждой карты
    MIRAGE_CT_CHOICES = [
        ('mirage_ct_b_anchor', 'Опорник Б'),
        ('mirage_ct_short', 'Шорт'),
        ('mirage_ct_window', 'Окно'),
        ('mirage_ct_con', 'Кон'),
        ('mirage_ct_a_anchor', 'Опорник А'),
    ]

    DUST2_CT_CHOICES = [
        ('dust2_ct_long', 'Лонг'),
        ('dust2_ct_mid', 'Мид'),
        ('dust2_ct_short', 'Шорт'),
        ('dust2_ct_b', 'Б'),
        ('dust2_ct_rotate', 'Ротейт'),
    ]

    ANUBIS_CT_CHOICES = [
        ('anubis_ct_b_anchor', 'Опорник Б'),
        ('anubis_ct_con', 'Кон'),
        ('anubis_ct_mid', 'Мид'),
        ('anubis_ct_a_anchor', 'Опорник А'),
        ('anubis_ct_rotate', 'Ротейт'),
    ]

    NUKE_CT_CHOICES = [
        ('nuke_ct_outside', 'Улица'),
        ('nuke_ct_main', 'Мейн'),
        ('nuke_ct_a_anchor', 'Опорник А'),
        ('nuke_ct_rotate', 'Ротейт'),
        ('nuke_ct_ramp', 'Рамп'),
    ]

    ANCIENT_CT_CHOICES = [
        ('ancient_ct_b_anchor', 'Опорник Б'),
        ('ancient_ct_cave', 'Кейв'),
        ('ancient_ct_mid', 'Мид'),
        ('ancient_ct_donate', 'Донат'),
        ('ancient_ct_a_anchor', 'Опорник А'),
    ]

    INFERNO_CT_CHOICES = [
        ('inferno_ct_b_anchor', 'Опорник Б'),
        ('inferno_ct_rotate', 'Ротейт'),
        ('inferno_ct_long', 'Лонг'),
        ('inferno_ct_short', 'Шорт'),
        ('inferno_ct_aps', 'АПС'),
    ]

    OVERPASS_CT_CHOICES = [
        ('overpass_ct_b_anchor', 'Опорник Б'),
        ('overpass_ct_rotate', 'Ротейт'),
        ('overpass_ct_mid', 'Мид'),
        ('overpass_ct_con', 'Кон'),
    ]

    # T роли
    T_ROLE_CHOICES = [
        ('', '-- Выберите роль --'),
        ('lurker', 'Люркер'),
        ('entry', 'Ентри'),
        ('support', 'Сапорт'),
    ]

    # Поля формы
    mirage_ct_position = forms.MultipleChoiceField(
        choices=MIRAGE_CT_CHOICES,
        required=False,
        label='Mirage (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    dust2_ct_position = forms.MultipleChoiceField(
        choices=DUST2_CT_CHOICES,
        required=False,
        label='Dust 2 (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    anubis_ct_position = forms.MultipleChoiceField(
        choices=ANUBIS_CT_CHOICES,
        required=False,
        label='Anubis (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    nuke_ct_position = forms.MultipleChoiceField(
        choices=NUKE_CT_CHOICES,
        required=False,
        label='Nuke (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    ancient_ct_position = forms.MultipleChoiceField(
        choices=ANCIENT_CT_CHOICES,
        required=False,
        label='Ancient (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    inferno_ct_position = forms.MultipleChoiceField(
        choices=INFERNO_CT_CHOICES,
        required=False,
        label='Inferno (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    overpass_ct_position = forms.MultipleChoiceField(
        choices=OVERPASS_CT_CHOICES,
        required=False,
        label='Overpass (CT)',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'map-options'})
    )

    t_role = forms.ChoiceField(
        choices=T_ROLE_CHOICES,
        required=False,
        label='Роль на T стороне',
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    is_igl = forms.BooleanField(
        required=False,
        label='Могу быть IGL (In-Game Leader)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    can_awp = forms.BooleanField(
        required=False,
        label='Могу играть на AWP',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    description = forms.CharField(
        required=False,
        label='Дополнительная информация',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Расскажите о себе, вашем опыте, стиле игры...',
            'rows': 4
        })
    )

    class Meta:
        model = PlayerProfile
        fields = [
            'mirage_ct_position', 'dust2_ct_position', 'anubis_ct_position',
            'nuke_ct_position', 'ancient_ct_position', 'inferno_ct_position',
            'overpass_ct_position', 't_role', 'is_igl', 'can_awp', 'description'
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ct_fields = [
            'mirage_ct_position',
            'dust2_ct_position',
            'anubis_ct_position',
            'nuke_ct_position',
            'ancient_ct_position',
            'inferno_ct_position',
            'overpass_ct_position',
        ]
        for field_name in ct_fields:
            stored_value = getattr(self.instance, field_name, '')
            if stored_value:
                self.fields[field_name].initial = [value for value in stored_value.split(',') if value]

    def save(self, commit=True):
        instance = super().save(commit=False)
        ct_fields = [
            'mirage_ct_position',
            'dust2_ct_position',
            'anubis_ct_position',
            'nuke_ct_position',
            'ancient_ct_position',
            'inferno_ct_position',
            'overpass_ct_position',
        ]
        for field_name in ct_fields:
            values = self.cleaned_data.get(field_name, [])
            if isinstance(values, (list, tuple)):
                setattr(instance, field_name, ','.join(values))
            else:
                setattr(instance, field_name, values or '')

        if commit:
            instance.save()
        return instance


class TeamRegistrationForm(UserCreationForm):
    """Форма регистрации команды"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Введите email команды'
        })
    )

    team_name = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Название команды'
        })
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'password1', 'password2')

        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Имя пользователя команды на сайте'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Пароль'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Подтверждение пароля'})

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError('Этот email уже используется')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.user_type = CustomUser.USER_TYPE_TEAM

        if commit:
            user.save()

        return user


class TeamDetailsForm(forms.ModelForm):
    """Форма деталей команды"""
    description = forms.CharField(
        required=True,
        label='Краткое описание команды',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Опишите вашу команду, цели, достижения...',
            'rows': 4
        })
    )

    # Требования к игрокам
    looking_for_igl = forms.BooleanField(
        required=False,
        label='Ищем IGL (In-Game Leader)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    looking_for_awper = forms.BooleanField(
        required=False,
        label='Ищем AWP-иста',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    class Meta:
        model = TeamProfile
        fields = ['description', 'looking_for_igl', 'looking_for_awper']
