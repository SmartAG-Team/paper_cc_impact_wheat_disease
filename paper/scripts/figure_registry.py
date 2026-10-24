"""Stable scientific roles keep manuscript and plotting-data numbering together."""
MAIN_ROLES = (
    'workflow', 'data_overview', 'attribution_design', 'monitoring',
    'climate_change', 'latitude_environment', 'climate_phenology',
    'attribution_results', 'production_exposure', 'adaptation',
)
SUPPLEMENT_ROLES = ('diagnostics', 'support', 'yield_context')
ROLE_NAMES = {role: f'fig{i}' for i, role in enumerate(MAIN_ROLES, 1)}
ROLE_NAMES.update({role: f'figs{i}' for i, role in enumerate(SUPPLEMENT_ROLES, 1)})
ROLE_NAMES['graphical_abstract'] = 'graphical_abstract'
AGREEMENT_COLORS={'increase':'#b2182b','decrease':'#2166ac',
    'mixed_or_zero':'#dfbf64','unavailable':'#d4d7da'}


def figure_number(role):
    name = ROLE_NAMES[role]
    return 'S'+name[4:] if name.startswith('figs') else name[3:]
