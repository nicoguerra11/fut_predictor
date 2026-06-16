import { useState } from 'react'
import ResultDisplay from './ResultDisplay'
import TeamSelect from './TeamSelect'
import styles from './MatchPredictor.module.css'

export default function MatchPredictor({ teams, apiBase }) {
  const [teamHome, setTeamHome] = useState('')
  const [teamAway, setTeamAway] = useState('')
  const [neutral, setNeutral] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const canPredict = teamHome && teamAway && teamHome !== teamAway && !loading

  const teamHomeObj = teams.find((t) => t.name === teamHome) || null
  const teamAwayObj = teams.find((t) => t.name === teamAway) || null

  function handlePredict() {
    if (!canPredict) return
    setLoading(true)
    setError(null)
    setResult(null)

    fetch(`${apiBase}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team_home: teamHome, team_away: teamAway, neutral }),
    })
      .then((r) => {
        if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || `Error HTTP ${r.status}`) })
        return r.json()
      })
      .then((data) => setResult(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Predecir partido</h2>
      <p className={styles.subtitle}>
        Seleccioná dos selecciones para ver las probabilidades según el modelo bayesiano.
      </p>

      {/* ── Selectores de equipos ─────────────────────────────────────── */}
      <div className={styles.teamsRow}>
        <div className={styles.teamBlock}>
          <span className={styles.label}>Local</span>
          <TeamSelect
            teams={teams}
            value={teamHome}
            onChange={(name) => { setTeamHome(name); setResult(null) }}
          />
          {teamHomeObj && (
            <div className={styles.rankPill}>
              🏆 Ranking FIFA #{teamHomeObj.fifaRanking}
            </div>
          )}
        </div>

        <div className={styles.vsLabel}>VS</div>

        <div className={styles.teamBlock}>
          <span className={styles.label}>Visitante</span>
          <TeamSelect
            teams={teams}
            value={teamAway}
            onChange={(name) => { setTeamAway(name); setResult(null) }}
          />
          {teamAwayObj && (
            <div className={styles.rankPill}>
              🏆 Ranking FIFA #{teamAwayObj.fifaRanking}
            </div>
          )}
        </div>
      </div>

      {/* ── Toggle sede neutral ──────────────────────────────────────── */}
      <label className={styles.neutralToggle}>
        <span
          className={`${styles.toggle} ${neutral ? styles.toggleOn : ''}`}
          onClick={() => setNeutral((v) => !v)}
          role="switch"
          aria-checked={neutral}
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setNeutral((v) => !v)}
        />
        <span className={styles.toggleLabel}>Sede neutral (sin ventaja de local)</span>
      </label>

      {/* ── Botón predecir ───────────────────────────────────────────── */}
      <button
        className={styles.predictBtn}
        disabled={!canPredict}
        onClick={handlePredict}
      >
        {loading ? 'Calculando…' : 'Predecir partido'}
      </button>

      {/* ── Skeleton loader ──────────────────────────────────────────── */}
      {loading && (
        <div className={styles.skeleton} aria-busy="true">
          <div className={styles.skeletonRow} />
          <div className={styles.skeletonCards}>
            <div className={styles.skeletonCard} />
            <div className={styles.skeletonCard} />
            <div className={styles.skeletonCard} />
          </div>
          <div className={styles.skeletonBar} />
          <div className={styles.skeletonBar} style={{ width: '80%' }} />
          <div className={styles.skeletonBar} style={{ width: '60%' }} />
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && !loading && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Resultado ────────────────────────────────────────────────── */}
      {result && !loading && (
        <ResultDisplay
          result={result}
          teamHomeObj={teamHomeObj}
          teamAwayObj={teamAwayObj}
        />
      )}
    </div>
  )
}
