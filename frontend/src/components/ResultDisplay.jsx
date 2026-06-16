import { useEffect, useRef } from 'react'
import ScorelineDisplay from './ScorelineDisplay'
import styles from './ResultDisplay.module.css'

function outcomeLabel(home, away) {
  if (home > away) return 'Victoria local'
  if (home === away) return 'Empate'
  return 'Victoria visitante'
}

export default function ResultDisplay({ result, teamHomeObj, teamAwayObj }) {
  const homeBarRef = useRef(null)
  const drawBarRef = useRef(null)
  const awayBarRef = useRef(null)

  const { prob_home, prob_draw, prob_away } = result
  const maxProb = Math.max(prob_home, prob_draw, prob_away)

  useEffect(() => {
    // Animar barras con CSS transition 0.6s
    const timeout = setTimeout(() => {
      if (homeBarRef.current) homeBarRef.current.style.width = `${(prob_home * 100).toFixed(1)}%`
      if (drawBarRef.current) drawBarRef.current.style.width = `${(prob_draw * 100).toFixed(1)}%`
      if (awayBarRef.current) awayBarRef.current.style.width = `${(prob_away * 100).toFixed(1)}%`
    }, 60)
    return () => clearTimeout(timeout)
  }, [prob_home, prob_draw, prob_away])

  const probCards = [
    { label: result.team_home, prob: prob_home, id: 'home' },
    { label: 'Empate',         prob: prob_draw, id: 'draw' },
    { label: result.team_away, prob: prob_away, id: 'away' },
  ]
  const barRefs = [homeBarRef, drawBarRef, awayBarRef]

  return (
    <div className={styles.container}>
      {/* ── Sección 1: Probabilidades ──────────────────────────────── */}
      <h3 className={styles.sectionTitle}>Probabilidades de resultado</h3>

      {/* Tres cards */}
      <div className={styles.probCards}>
        {probCards.map(({ label, prob, id }) => {
          const isFav = prob === maxProb
          return (
            <div key={id} className={`${styles.probCard} ${isFav ? styles.probCardFav : ''}`}>
              <div className={styles.probCardLabel}>{label}</div>
              <div className={styles.probCardValue}>{(prob * 100).toFixed(1)}%</div>
            </div>
          )
        })}
      </div>

      {/* Barras animadas */}
      <div className={styles.bars}>
        {probCards.map(({ label, prob }, idx) => (
          <div key={label} className={styles.barRow}>
            <span className={styles.barLabel}>{label}</span>
            <div className={styles.barTrack}>
              <div
                ref={barRefs[idx]}
                className={styles.bar}
                style={{ width: '0%', transition: 'width 0.6s ease' }}
              />
            </div>
            <span className={styles.barPct}>{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* Cards de goles esperados */}
      <div className={styles.goalsCards}>
        <div className={styles.goalsCard}>
          <div className={styles.goalsFlag}>{teamHomeObj?.flag ?? '🏴'}</div>
          <div className={styles.goalsTeam}>{result.team_home}</div>
          <div className={styles.goalsValue}>
            {result.expected_goals_home.toFixed(1)}
          </div>
          <div className={styles.goalsCI}>
            IC 89%: {result.credible_interval_home[0]}–{result.credible_interval_home[1]} goles
          </div>
        </div>
        <div className={styles.goalsCard}>
          <div className={styles.goalsFlag}>{teamAwayObj?.flag ?? '🏴'}</div>
          <div className={styles.goalsTeam}>{result.team_away}</div>
          <div className={styles.goalsValue}>
            {result.expected_goals_away.toFixed(1)}
          </div>
          <div className={styles.goalsCI}>
            IC 89%: {result.credible_interval_away[0]}–{result.credible_interval_away[1]} goles
          </div>
        </div>
      </div>

      <p className={styles.simNote}>
        Basado en 4.000 simulaciones desde la posterior bayesiana
      </p>

      {/* ── Sección 2: Marcadores exactos ─────────────────────────── */}
      <ScorelineDisplay
        scorelines={result.scorelines}
        teamHome={result.team_home}
        teamAway={result.team_away}
      />
    </div>
  )
}
