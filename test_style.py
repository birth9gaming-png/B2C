"""Tests for the 'your style' copy-of-you model (offline, simulated games)."""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from test_backend import simulated


def fake_picks(games, league, n_weeks=8):
    """A bettor who always takes home favorites of 3-10: moneyline (CFB) or alt spread, else the spread."""
    done = games[games.completed & games.spread_home.notna()]
    weeks = sorted(set(zip(done.season, done.week)))[-n_weeks:]
    rows = []
    for s, w in weeks:
        g = done[(done.season == s) & (done.week == w)]
        for r in g.itertuples():
            if -10 <= r.spread_home <= -3:
                kind = 'moneyline' if league == 'CFB' else 'alt spread'
                line = np.nan if kind == 'moneyline' else r.spread_home + 2
                margin = r.home_score - r.away_score
                res = 'W' if (margin > 0 if kind == 'moneyline' else margin + line > 0) else 'L'
                rows.append(dict(placed=r.kickoff - pd.Timedelta(hours=3), league=league, season=s, week=w, game_id=r.id,
                                 away=r.away, home=r.home, team=r.home, side='home', kind=kind, line=line,
                                 neutral=False, bet_type='parlay', result=res))
    return pd.DataFrame(rows)


class TestStyle(unittest.TestCase):
    def test_learns_and_suggests(self):
        for lg, now in (('NFL', '2025-11-01T12:00:00Z'), ('CFB', '2025-10-15T12:00:00Z')):
            data, t = simulated(lg, now)
            g = data['games']
            picks = fake_picks(g, lg)
            self.assertGreater(len(picks), style.MIN_PICKS)
            m = style.train(picks, g, lg)
            self.assertIsNotNone(m)
            coef = dict(zip(m['feats'], m['coef']))
            self.assertGreater(coef['home'], 0)                      # learned "home teams"
            self.assertGreater(coef['b2'], 0)                        # learned "favorites of 3-7"
            leg = style.how_you_play(m, -5.0, lg)
            self.assertEqual(leg['kind'], 'moneyline' if lg == 'CFB' else 'alt spread')
            if lg == 'NFL':
                self.assertNotIn('moneyline', m['forms']['2']['counts'])
            # this week's board: upcoming games with a current line
            up = g[(~g.completed) & (g.kickoff > t) & (g.kickoff < t + pd.Timedelta(days=8))].copy()
            board = up.assign(current_home=up.spread_home, open_home=up.spread_home, edge=0.0,
                              injuries=[{} for _ in range(len(up))])
            sug = style.suggest(m, board, g, lg, picks=picks, k=3)
            self.assertEqual(int(sug.shortlist.sum()), min(3, len(sug)))
            self.assertTrue(sug['raw'].is_monotonic_decreasing)
            top = sug[sug.shortlist].iloc[0]
            self.assertTrue(0 <= top['style'] <= 100)
            rep = style.replay(m, g, lg, sorted(g.season.unique())[:-1], k=3)
            self.assertGreater(sum(rep['legs']), 0)

    def test_leg_text_and_grading(self):
        self.assertEqual(style.leg_text('Alabama', dict(kind='moneyline', leg_line=None)), 'Alabama ML')
        self.assertEqual(style.leg_text('KC', dict(kind='alt spread', leg_line=-2.5)), 'KC -2.5 (alt)')
        self.assertEqual(style.grade_leg(dict(kind='alt spread', leg_line=-2.5), 3), 'W')
        self.assertEqual(style.grade_leg(dict(kind='spread', leg_line=-7.5), 7), 'L')
        self.assertEqual(style.grade_leg(dict(kind='moneyline', leg_line=None), 1), 'W')
        self.assertEqual(style.bucket_name(-5), 'favorite 3-7')
        self.assertEqual(style.bucket_name(10), 'underdog 7+')


if __name__ == '__main__':
    unittest.main()
