from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q
from django.utils import timezone

from .forms import (
    RegistrationStep1Form,
    PlayerRegistrationForm,
    PlayerDetailsForm,
    TeamRegistrationForm,
    TeamDetailsForm
)
from .models import CustomUser, PlayerProfile, TeamProfile
from .faceit import check_faceit_nickname, fetch_faceit_profile_details


# ---------------------------
# HOME
# ---------------------------
def home(request):
    """Главная страница"""
    context = {}

    if request.user.is_authenticated:
        if request.user.user_type == 'player':
            context['feed_title'] = 'Команды, которые ищут игроков'
            context['feed_items'] = (
                TeamProfile.objects.select_related('user')
                .filter(is_active=True)
                .order_by('-id')[:5]
            )
            context['feed_kind'] = 'team'
        else:
            context['feed_title'] = 'Игроки, которые готовы присоединиться'
            context['feed_items'] = (
                PlayerProfile.objects.select_related('user')
                .order_by('-id')[:5]
            )
            context['feed_kind'] = 'player'

    return render(request, 'users/home.html', context)


# ---------------------------
# REGISTRATION STEP 1 - Выбор типа
# ---------------------------
def register_step1(request):
    """Первый шаг регистрации - выбор типа аккаунта"""
    if request.method == 'POST':
        form = RegistrationStep1Form(request.POST)
        if form.is_valid():
            user_type = form.cleaned_data['user_type']
            request.session['register_user_type'] = user_type
            return redirect('register_step2')
    else:
        form = RegistrationStep1Form()

    return render(request, 'users/register_step1.html', {'form': form})


# ---------------------------
# REGISTRATION STEP 2 - Основные данные
# ---------------------------
def register_step2(request):
    """Второй шаг регистрации - основные данные"""
    user_type = request.session.get('register_user_type')

    if not user_type:
        return redirect('register_step1')

    if user_type == 'player':
        form_class = PlayerRegistrationForm
        template = 'users/register_player.html'
    else:
        form_class = TeamRegistrationForm
        template = 'users/register_team.html'

    if request.method == 'POST':
        form = form_class(request.POST)

        if form.is_valid():
            user = form.save()

            # Сохраняем данные в сессии для следующего шага
            request.session['register_user_id'] = user.id

            if user_type == 'player':
                # Загружаем данные с Faceit
                try:
                    player_profile = PlayerProfile.objects.create(user=user)
                    faceit_data = fetch_faceit_profile_details(user.faceit_nickname)

                    if faceit_data:
                        player_profile.faceit_player_id = faceit_data.get('player_id', '')
                        player_profile.country = faceit_data.get('country', '')
                        player_profile.level = faceit_data.get('skill_level', 0)
                        player_profile.elo = faceit_data.get('faceit_elo', 0)
                        player_profile.skill_level = faceit_data.get('skill_level', 0)
                        player_profile.faceit_url = faceit_data.get('faceit_url', '')
                        player_profile.avatar = faceit_data.get('avatar', '')
                        player_profile.matches = faceit_data.get('matches', 0)
                        player_profile.wins = faceit_data.get('wins', 0)
                        player_profile.win_rate = faceit_data.get('win_rate', 0.0)
                        player_profile.average_kd = faceit_data.get('average_kd', 0.0)
                        player_profile.average_hs = faceit_data.get('average_hs', 0.0)
                        player_profile.current_win_streak = faceit_data.get('current_win_streak', 0)
                        player_profile.longest_win_streak = faceit_data.get('longest_win_streak', 0)
                        player_profile.save()

                        messages.success(request, f'Данные с Faceit успешно загружены! Уровень: {player_profile.level}')
                except Exception as e:
                    messages.warning(request, f'Данные с Faceit загружены частично: {str(e)}')

                return redirect('register_step3')
            else:
                return redirect('register_step3')

        return render(request, template, {'form': form})

    form = form_class()
    return render(request, template, {'form': form})


# ---------------------------
# REGISTRATION STEP 3 - Детали
# ---------------------------
def register_step3(request):
    """Третий шаг регистрации - детали профиля"""
    user_type = request.session.get('register_user_type')
    user_id = request.session.get('register_user_id')

    if not user_type or not user_id:
        return redirect('register_step1')

    user = get_object_or_404(CustomUser, id=user_id)

    if user_type == 'player':
        try:
            player_profile = PlayerProfile.objects.get(user=user)
        except PlayerProfile.DoesNotExist:
            player_profile = PlayerProfile.objects.create(user=user)

        if request.method == 'POST':
            form = PlayerDetailsForm(request.POST, instance=player_profile)
            if form.is_valid():
                form.save()

                # Логиним пользователя
                login(request, user)

                # Очищаем сессию
                request.session.pop('register_user_type', None)
                request.session.pop('register_user_id', None)

                messages.success(request, 'Регистрация игрока успешно завершена!')
                return redirect('profile', user_id=user.id)
        else:
            form = PlayerDetailsForm(instance=player_profile)

        template = 'users/register_player_details.html'

    else:  # team
        try:
            team_profile = TeamProfile.objects.get(user=user)
        except TeamProfile.DoesNotExist:
            team_profile = TeamProfile.objects.create(user=user, team_name=user.username)

        if request.method == 'POST':
            form = TeamDetailsForm(request.POST, instance=team_profile)
            if form.is_valid():
                team_profile = form.save(commit=False)
                team_profile.user = user
                team_profile.save()

                # Логиним пользователя
                login(request, user)

                # Очищаем сессию
                request.session.pop('register_user_type', None)
                request.session.pop('register_user_id', None)

                messages.success(request, 'Регистрация команды успешно завершена!')
                return redirect('profile', user_id=user.id)
        else:
            form = TeamDetailsForm(instance=team_profile)

        template = 'users/register_team_details.html'

    return render(request, template, {
        'form': form,
        'user': user,
        'user_type': user_type
    })


# ---------------------------
# REGISTER (основная функция для ссылок)
# ---------------------------
def register(request):
    """Основная точка входа для регистрации"""
    return redirect('register_step1')


# ---------------------------
# PROFILE
# ---------------------------
@login_required
def profile(request, user_id):
    """Страница профиля пользователя"""
    user = get_object_or_404(CustomUser, id=user_id)

    # Проверяем права доступа
    if not request.user.is_superuser and request.user != user:
        messages.error(request, 'У вас нет доступа к этому профилю.')
        return redirect('home')

    context = {'profile_user': user, 'steamid64': ''}

    if user.user_type == CustomUser.USER_TYPE_PLAYER:
        try:
            player_profile = PlayerProfile.objects.get(user=user)
            context['player_profile'] = player_profile
            context['steamid64'] = (
                getattr(player_profile, "steamid64", None)
                or getattr(player_profile, "steam_id64", None)
                or getattr(player_profile, "steam_id", None)
                or ""
            )
            context['steamid64_debug'] = {
                "steamid64": getattr(player_profile, "steamid64", None),
                "steam_id64": getattr(player_profile, "steam_id64", None),
                "steam_id": getattr(player_profile, "steam_id", None),
            }
        except PlayerProfile.DoesNotExist:
            messages.warning(request, 'Профиль игрока не найден.')
            context['player_profile'] = None

    else:
        try:
            team_profile = TeamProfile.objects.get(user=user)
            context['team_profile'] = team_profile
        except TeamProfile.DoesNotExist:
            messages.warning(request, 'Профиль команды не найден.')
            context['team_profile'] = None

    return render(request, 'users/profile.html', context)


# ---------------------------
# EDIT PROFILE
# ---------------------------
@login_required
def edit_profile(request):
    """Редактирование профиля (для игроков и команд)"""
    user = request.user

    if user.user_type == CustomUser.USER_TYPE_PLAYER:
        # Редактирование профиля игрока
        try:
            player_profile = PlayerProfile.objects.get(user=user)
        except PlayerProfile.DoesNotExist:
            player_profile = PlayerProfile.objects.create(user=user)

        if request.method == 'POST':
            form = PlayerDetailsForm(request.POST, instance=player_profile)
            if form.is_valid():
                form.save()
                messages.success(request, 'Профиль игрока успешно обновлен!')
                return redirect('profile', user_id=user.id)
        else:
            form = PlayerDetailsForm(instance=player_profile)

        template = 'users/edit_player_profile.html'
        context = {'form': form, 'user_type': 'player'}

    else:
        # Редактирование профиля команды
        try:
            team_profile = TeamProfile.objects.get(user=user)
        except TeamProfile.DoesNotExist:
            team_profile = TeamProfile.objects.create(user=user, team_name=user.username)

        if request.method == 'POST':
            form = TeamDetailsForm(request.POST, instance=team_profile)
            if form.is_valid():
                form.save()
                messages.success(request, 'Профиль команды успешно обновлен!')
                return redirect('profile', user_id=user.id)
        else:
            form = TeamDetailsForm(instance=team_profile)

        template = 'users/edit_team_profile.html'
        context = {'form': form, 'user_type': 'team', 'user': user}

    return render(request, template, context)


# ---------------------------
# UPDATE FACEIT DATA
# ---------------------------
@login_required
def update_faceit_data(request):
    """Обновить данные профиля с Faceit (только для игроков)"""
    if request.user.user_type != CustomUser.USER_TYPE_PLAYER:
        messages.error(request, 'Эта функция доступна только для игроков.')
        return redirect('profile', user_id=request.user.id)

    try:
        player_profile = PlayerProfile.objects.get(user=request.user)

        if not request.user.faceit_nickname:
            messages.error(request, 'У вас не указан Faceit никнейм.')
            return redirect('profile', user_id=request.user.id)

        # Получаем данные с Faceit
        faceit_data = fetch_faceit_profile_details(request.user.faceit_nickname)

        if faceit_data:
            player_profile.faceit_player_id = faceit_data.get('player_id', player_profile.faceit_player_id)
            player_profile.country = faceit_data.get('country', player_profile.country)
            player_profile.level = faceit_data.get('skill_level', player_profile.level)
            player_profile.elo = faceit_data.get('faceit_elo', player_profile.elo)
            player_profile.skill_level = faceit_data.get('skill_level', player_profile.skill_level)
            player_profile.faceit_url = faceit_data.get('faceit_url', player_profile.faceit_url)
            player_profile.avatar = faceit_data.get('avatar', player_profile.avatar)
            player_profile.matches = faceit_data.get('matches', player_profile.matches)
            player_profile.wins = faceit_data.get('wins', player_profile.wins)
            player_profile.win_rate = faceit_data.get('win_rate', player_profile.win_rate)
            player_profile.average_kd = faceit_data.get('average_kd', player_profile.average_kd)
            player_profile.average_hs = faceit_data.get('average_hs', player_profile.average_hs)
            player_profile.current_win_streak = faceit_data.get('current_win_streak', player_profile.current_win_streak)
            player_profile.longest_win_streak = faceit_data.get('longest_win_streak', player_profile.longest_win_streak)
            player_profile.last_faceit_update = timezone.now()
            player_profile.save()

            messages.success(request,
                             f'Данные успешно обновлены! Уровень: {player_profile.level}, ELO: {player_profile.elo}')
        else:
            messages.error(request, 'Не удалось получить данные с Faceit.')

    except PlayerProfile.DoesNotExist:
        messages.error(request, 'Профиль игрока не найден.')

    return redirect('profile', user_id=request.user.id)


# ---------------------------
# SEARCH PLAYERS (только для команд)
# ---------------------------
@login_required
def search_player(request):
    """Поиск игроков (доступно только командам)"""
    if request.user.user_type != CustomUser.USER_TYPE_TEAM:
        messages.error(request, 'Поиск игроков доступен только командам.')
        return redirect('home')

    players = PlayerProfile.objects.select_related('user').all()

    position_filters = []

    # Фильтры
    filters = {}

    # Уровень
    min_level = request.GET.get('min_level')
    max_level = request.GET.get('max_level')

    if min_level:
        try:
            filters['level__gte'] = int(min_level)
        except ValueError:
            pass

    if max_level:
        try:
            filters['level__lte'] = int(max_level)
        except ValueError:
            pass

    # ELO
    min_elo = request.GET.get('min_elo')
    max_elo = request.GET.get('max_elo')

    if min_elo:
        try:
            filters['elo__gte'] = int(min_elo)
        except ValueError:
            pass

    if max_elo:
        try:
            filters['elo__lte'] = int(max_elo)
        except ValueError:
            pass

    # IGL
    need_igl = request.GET.get('igl')
    if need_igl == 'true':
        filters['is_igl'] = True

    # AWP
    need_awp = request.GET.get('awp')
    if need_awp == 'true':
        filters['can_awp'] = True

    # Позиции на картах
    map_labels = dict(PlayerProfile.MAP_CHOICES)
    for map_name, choices in PlayerProfile.CT_POSITION_CHOICES.items():
        selected = request.GET.getlist(f'{map_name}_positions')
        if selected:
            map_query = Q()
            for position in selected:
                map_query |= Q(**{f'{map_name}_ct_position__contains': position})
            players = players.filter(map_query)
        position_filters.append({
            'map_name': map_name,
            'label': map_labels.get(map_name, map_name.title()),
            'choices': choices,
            'selected': selected,
        })

    # Страна
    country = request.GET.get('country')
    if country:
        filters['country__iexact'] = country

    # Применяем фильтры
    if filters:
        players = players.filter(**filters)

    # Сортировка
    sort_by = request.GET.get('sort_by', 'level')
    if sort_by in ['level', 'elo', 'win_rate', 'average_kd']:
        players = players.order_by(f'-{sort_by}')

    context = {
        'players': players,
        'position_filters': position_filters,
        'filters': {
            'min_level': min_level,
            'max_level': max_level,
            'min_elo': min_elo,
            'max_elo': max_elo,
            'igl': need_igl,
            'awp': need_awp,
            'country': country,
            'sort_by': sort_by,
        }
    }

    return render(request, 'users/search_player.html', context)


# ---------------------------
# SEARCH TEAMS (только для игроков)
# ---------------------------
@login_required
def search_team(request):
    """Поиск команд (доступно только игрокам)"""
    if request.user.user_type != CustomUser.USER_TYPE_PLAYER:
        messages.error(request, 'Поиск команд доступен только игрокам.')
        return redirect('home')

    teams = TeamProfile.objects.filter(is_active=True).select_related('user')

    # Фильтры
    need_igl = request.GET.get('igl')
    need_awper = request.GET.get('awper')

    if need_igl == 'true':
        teams = teams.filter(looking_for_igl=True)

    if need_awper == 'true':
        teams = teams.filter(looking_for_awper=True)

    context = {
        'teams': teams,
        'filters': {
            'igl': need_igl,
            'awper': need_awper,
        }
    }

    return render(request, 'users/search_team.html', context)


# ---------------------------
# PLAYER DETAILS
# ---------------------------
@login_required
def player_details(request, player_id):
    """Детальная информация об игроке (только для команд)"""
    if request.user.user_type != CustomUser.USER_TYPE_TEAM:
        messages.error(request, 'Эта страница доступна только командам.')
        return redirect('home')

    player_profile = get_object_or_404(PlayerProfile, id=player_id)

    context = {
        'player_profile': player_profile,
        'profile_user': player_profile.user,
    }

    return render(request, 'users/player_details.html', context)


# ---------------------------
# TEAM DETAILS
# ---------------------------
@login_required
def team_details(request, team_id):
    """Детальная информация о команде (только для игроков)"""
    if request.user.user_type != CustomUser.USER_TYPE_PLAYER:
        messages.error(request, 'Эта страница доступна только игрокам.')
        return redirect('home')

    team_profile = get_object_or_404(TeamProfile, id=team_id)

    context = {
        'team_profile': team_profile,
        'profile_user': team_profile.user,
    }

    return render(request, 'users/team_details.html', context)
