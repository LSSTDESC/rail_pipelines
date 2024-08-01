import os

import pytest
from rail.utils import name_utils

def test_name_utils():
    
    assert name_utils._get_required_interpolants('xx_{alice}_{bob}') == ['{alice}', '{bob}']
    assert name_utils._format_template('xx_{alice}_{bob}', alice='x', bob='x') == 'xx_x_x'

    test_dict = dict(
        a='a_{alice}',
        #b=['b1_{alice}', 'b2_{bob}'],
        c=dict(c1='c1_{alice}', c2='c2_{alice}'),
        #c2=dict(c2_1='c1_{alice}', c2_2=['c2_{alice}', 'c2_{bob}']),
    )

    name_utils._resolve_dict(test_dict, dict(alice='x', bob='y'))

    assert name_utils._resolve_dict(None, {}) is None
    with pytest.raises(ValueError):
        name_utils._resolve_dict(dict(a=('s','d',)), dict(alice='x', bob='y'))


    
