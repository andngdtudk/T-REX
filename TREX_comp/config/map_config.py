# 'warmup': 100 for every network -- manuscript Section 3.5: "Each traffic episode
# simulates 3,600 seconds, including a 100-second warm-up phase." Confirmed against the
# manuscript text directly by the repo owner (previously flagged as `warmup=0`, a
# code-vs-brief discrepancy; see AUDIT_REPORT.md). Applies uniformly: every network below
# has an `end_time - start_time` of exactly 3600s, matching the manuscript's blanket
# per-episode duration, so there's no network-specific exception to carve out.
map_configs = {
    'grid4x4': {
        'lights': [],
        'net': 'environments/grid4x4/grid4x4.net.xml',
        'route': 'environments/grid4x4/grid4x4',
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 0,
        'end_time': 3600,
        'warmup': 100
    },
    'arterial4x4': {
        'lights': [],
        'net': 'environments/arterial4x4/arterial4x4.net.xml',
        'route': 'environments/arterial4x4/arterial4x4',
        'step_length': 5,
        'yellow_length': 2,
        'step_ratio': 1,
        'start_time': 0,
        'end_time': 3600,
        'warmup': 100
    },
    'arterial5x5': {
        'lights': [],
        'net': 'environments/arterial5x5/exp.sumocfg',
        'route': None,
        'step_length': 5,
        'yellow_length': 2,
        'step_ratio': 1,
        'start_time': 0,
        'end_time': 3600,
        'warmup': 100
    },
    'ingolstadt1': {
        'lights': [],
        'net': 'environments/ingolstadt1/ingolstadt1.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 57600,
        'end_time': 61200,
        'warmup': 100
    },
    'ingolstadt7': {
        'lights': ['cluster_1757124350_1757124352',
                   'gneJ143', 'gneJ207',
                   'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190',
                   '32564122', 'gneJ260', 'gneJ210'
                   ],
        'net': 'environments/ingolstadt7/ingolstadt7.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 57600,
        'end_time': 61200,
        'warmup': 100
    },
    'ingolstadt21': {
        'lights': ['1863241632', '2330725114', '243351999', '243641585', '243749571', '30503246', '30624898', '32564122', '89127267', '89173763', '89173808', 'cluster_1427494838_273472399', 'cluster_1757124350_1757124352', 'cluster_1863241547_1863241548_1976170214', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190', 'gneJ143', 'gneJ207', 'gneJ208', 'gneJ210', 'gneJ255', 'gneJ257'],
        'net': 'environments/ingolstadt21/ingolstadt21.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 57600,
        'end_time': 61200,
        'warmup': 100
    },
    'cologne1': {
        'lights': [],
        'net': 'environments/cologne1/cologne1.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 25200,
        'end_time': 28800,
        'warmup': 100
    },
    'cologne3': {
        'lights': [],
        'net': 'environments/cologne3/cologne3.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 25200,
        'end_time': 28800,
        'warmup': 100
    },
    'cologne8': {
        'lights': [],
        'net': 'environments/cologne8/cologne8.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 25200,
        'end_time': 28800,
        'warmup': 100
    },
    'turin5': {
        'lights': [],
        'net': 'environments/turin5/turin5.sumocfg',
        'route': None,
        'step_length': 10,
        'yellow_length': 3,
        'step_ratio': 1,
        'start_time': 65400,
        'end_time': 69000,
        'warmup': 100
    },
}
