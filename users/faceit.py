import requests
import logging
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


class FaceitAPI:
    def __init__(self):
        self.api_key = settings.FACEIT_API_KEY
        self.base_url = settings.FACEIT_API_URL
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }

    def get_player_by_nickname(self, nickname):
        """Получить информацию об игроке по никнейму"""
        url = f'{self.base_url}/players'
        params = {'nickname': nickname, 'game': 'cs2'}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Faceit player not found: {nickname}")
                return None
            else:
                logger.error(f"Faceit API error {response.status_code} for nickname {nickname}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Faceit API connection error for nickname {nickname}: {e}")
            return None

    def check_nickname_exists(self, nickname):
        """Проверить, существует ли никнейм на Faceit"""
        data = self.get_player_by_nickname(nickname)
        return data is not None and 'player_id' in data

    def get_player_stats(self, player_id):
        """Получить статистику игрока"""
        url = f'{self.base_url}/players/{player_id}/stats/cs2'

        try:
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Faceit stats not found for player: {player_id}")
                return {}
            else:
                logger.error(f"Faceit API stats error {response.status_code} for player {player_id}: {response.text}")
                return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Faceit API stats connection error for player {player_id}: {e}")
            return {}

    def get_player_details(self, nickname):
        """Получить детальную информацию об игроке для создания профиля"""
        player_data = self.get_player_by_nickname(nickname)
        if not player_data:
            return None

        player_id = player_data.get('player_id')

        # Базовые данные игрока
        details = {
            'player_id': player_id,
            'nickname': player_data.get('nickname'),
            'country': player_data.get('country', ''),
            'avatar': player_data.get('avatar', ''),
            'faceit_url': player_data.get('faceit_url', f'https://www.faceit.com/en/players/{nickname}'),
            'steam_id_64': player_data.get('steam_id_64', ''),
            'steam_nickname': player_data.get('steam_nickname', ''),
        }

        # CS2 статистика
        games = player_data.get('games', {})
        cs2_data = games.get('cs2', {})

        details.update({
            'skill_level': cs2_data.get('skill_level', 0),
            'faceit_elo': cs2_data.get('faceit_elo', 0),
            'game_player_id': cs2_data.get('game_player_id', ''),
        })

        # Дополнительные метрики из статистики
        stats = self.get_player_stats(player_id) if player_id else {}

        if stats and 'lifetime' in stats:
            lifetime = stats['lifetime']

            # Извлекаем статистику
            matches = lifetime.get('Matches', '0')
            wins = lifetime.get('Wins', '0')
            average_kd = lifetime.get('Average K/D Ratio', '0')
            average_hs = lifetime.get('Average Headshots %', '0')

            # Преобразуем строки в числа
            try:
                matches = int(matches) if matches else 0
                wins = int(wins) if wins else 0
                average_kd = float(average_kd) if average_kd else 0.0
                average_hs = float(average_hs.rstrip('%')) if average_hs else 0.0
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error parsing stats for player {player_id}: {e}")
                matches = wins = 0
                average_kd = average_hs = 0.0

            win_rate = (wins / matches * 100) if matches > 0 else 0

            details.update({
                'matches': matches,
                'wins': wins,
                'win_rate': round(win_rate, 1),
                'average_kd': round(average_kd, 2),
                'average_hs': round(average_hs, 1),
                'current_win_streak': lifetime.get('Current Win Streak', 0),
                'longest_win_streak': lifetime.get('Longest Win Streak', 0),
            })
        else:
            # Если статистики нет, используем значения по умолчанию
            details.update({
                'matches': 0,
                'wins': 0,
                'win_rate': 0.0,
                'average_kd': 0.0,
                'average_hs': 0.0,
                'current_win_streak': 0,
                'longest_win_streak': 0,
            })

        return details


# Создаем глобальный экземпляр API
faceit_api = FaceitAPI()


def check_faceit_nickname(nickname: str) -> bool:
    """Проверить существование Faceit никнейма"""
    return faceit_api.check_nickname_exists(nickname)


def fetch_faceit_profile_details(nickname: str) -> dict:
    """Получить детальную информацию с Faceit для создания профиля"""
    return faceit_api.get_player_details(nickname)