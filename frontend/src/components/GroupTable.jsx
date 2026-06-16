import { useState, useEffect } from 'react'
import styles from './GroupTable.module.css'

export default function GroupTable({ apiBase }) {
  const [standings, setStandings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${apiBase}/standings`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => setStandings(data.standings || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [apiBase])

  if (loading) return <div className={styles.loading}>Cargando tabla de grupos…</div>
  if (error) return <div className={styles.error}>Error al cargar: {error}</div>

  // Agrupar por grupo
  const groups = {}
  for (const entry of standings) {
    if (!groups[entry.group]) groups[entry.group] = []
    groups[entry.group].push(entry)
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Tabla de grupos</h2>
      <p className={styles.subtitle}>
        Puntos predichos con el modelo bayesiano para partidos pendientes.
      </p>

      <div className={styles.grid}>
        {Object.entries(groups).map(([group, teams]) => (
          <div key={group} className={styles.groupCard}>
            <div className={styles.groupHeader}>Grupo {group}</div>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Equipo</th>
                  <th title="Puntos reales">PTS</th>
                  <th title="Puntos predichos totales">Pred.</th>
                  <th title="Probabilidad de clasificar">P(clasif.)</th>
                </tr>
              </thead>
              <tbody>
                {teams.map((team, i) => (
                  <tr key={team.team} className={i < 2 ? styles.qualifier : ''}>
                    <td className={styles.teamCell}>{team.team}</td>
                    <td>{team.points}</td>
                    <td className={styles.predicted}>{team.predicted_points.toFixed(1)}</td>
                    <td>
                      <div className={styles.probCell}>
                        <div
                          className={styles.probBar}
                          style={{ width: `${(team.prob_advance * 100).toFixed(0)}%` }}
                        />
                        <span>{(team.prob_advance * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
