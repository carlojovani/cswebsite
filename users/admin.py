from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, PlayerProfile, TeamProfile


class PlayerProfileInline(admin.StackedInline):
    model = PlayerProfile
    can_delete = False
    verbose_name_plural = 'Player Profile'
    fk_name = 'user'
    fields = [
        'faceit_player_id', 'country', 'level', 'elo', 'skill_level',
        'matches', 'wins', 'win_rate', 'average_kd', 'average_hs',
        'is_igl', 'can_awp', 't_role', 'description'
    ]


class TeamProfileInline(admin.StackedInline):
    model = TeamProfile
    can_delete = False
    verbose_name_plural = 'Team Profile'
    fk_name = 'user'
    fields = ['team_name', 'description', 'looking_for_igl', 'looking_for_awper', 'is_active']


class CustomUserAdmin(UserAdmin):
    inlines = (PlayerProfileInline, TeamProfileInline)
    list_display = ('username', 'email', 'faceit_nickname', 'user_type', 'is_staff', 'registration_date')
    list_filter = ('user_type', 'is_staff', 'is_superuser', 'registration_date')
    search_fields = ('username', 'email', 'faceit_nickname')
    ordering = ('-registration_date',)

    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('email', 'faceit_nickname', 'user_type')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined', 'registration_date')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'faceit_nickname', 'user_type', 'password1', 'password2'),
        }),
    )

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return []

        if obj.user_type == 'player':
            return [PlayerProfileInline(self.model, self.admin_site)]
        elif obj.user_type == 'team':
            return [TeamProfileInline(self.model, self.admin_site)]

        return []


class PlayerProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'level', 'elo', 'country', 'is_igl', 'can_awp', 'last_faceit_update')
    list_filter = ('level', 'country', 'is_igl', 'can_awp')
    search_fields = ('user__username', 'user__faceit_nickname', 'faceit_player_id', 'steam_id')
    readonly_fields = ('last_faceit_update', 'profile_created')

    fieldsets = (
        ('User Info', {'fields': ('user',)}),
        ('Faceit Data', {'fields': ('faceit_player_id', 'last_faceit_update')}),
        ('Stats', {'fields': ('level', 'elo', 'skill_level', 'matches', 'wins', 'win_rate',
                              'average_kd', 'average_hs', 'current_win_streak', 'longest_win_streak')}),
        ('Game Info', {'fields': ('is_igl', 'can_awp', 't_role', 'description')}),
        ('CT Positions', {'fields': (
            'mirage_ct_position', 'dust2_ct_position', 'anubis_ct_position',
            'nuke_ct_position', 'ancient_ct_position', 'inferno_ct_position',
            'overpass_ct_position'
        ), 'classes': ('collapse',)}),
        ('Profile Info', {'fields': ('country', 'steam_id', 'avatar', 'faceit_url')}),
        ('Dates', {'fields': ('profile_created',)}),
    )


class TeamProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'team_name', 'looking_for_igl', 'looking_for_awper', 'is_active', 'created_date')
    list_filter = ('is_active', 'looking_for_igl', 'looking_for_awper', 'created_date')
    search_fields = ('user__username', 'team_name', 'description')
    readonly_fields = ('created_date',)

    fieldsets = (
        ('User Info', {'fields': ('user',)}),
        ('Team Info', {'fields': ('team_name', 'description')}),
        ('Looking For', {'fields': ('looking_for_igl', 'looking_for_awper')}),
        ('Status', {'fields': ('is_active', 'created_date')}),
    )


admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(PlayerProfile, PlayerProfileAdmin)
admin.site.register(TeamProfile, TeamProfileAdmin)