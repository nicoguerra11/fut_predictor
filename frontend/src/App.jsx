import { useState, useEffect } from 'react'
import MatchPredictor from './components/MatchPredictor'
import GroupTable from './components/GroupTable'
import { TEAMS, TEAM_LOOKUP } from './data/teams'
import styles from './styles/App.module.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [activeTab, setActiveTab] = useState('predictor')
  // Equipos disponibles en el modelo (filtrados por la API); fallback a todos los 48
  const [availableTeams, setAvailableTeams] = useState(TEAMS)
  const [apiError, setApiError] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/teams`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => {
        const modelTeamNames = new Set(data.teams || [])
        if (modelTeamNames.size > 0) {
          // Siempre mostrar los 48 clasificados al Mundial 2026 (teams.js).
          // Los equipos sin historial en 2014/2018/2022 usan el prior global del modelo.
          setAvailableTeams(TEAMS)
        }
      })
      .catch((err) => {
        // La API no está lista aún (modelo sin entrenar); se usa teams.js como fallback
        setApiError(err.message)
      })
  }, [])

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>⚽</span>
            <h1>Mundial 2026</h1>
            <span className={styles.badge}>FIFA World Cup 2026</span>
          </div>
          <nav className={styles.nav}>
            <button
              className={`${styles.navBtn} ${activeTab === 'predictor' ? styles.active : ''}`}
              onClick={() => setActiveTab('predictor')}
            >
              Predecir
            </button>
            <button
              className={`${styles.navBtn} ${activeTab === 'groups' ? styles.active : ''}`}
              onClick={() => setActiveTab('groups')}
            >
              Grupos
            </button>
          </nav>
        </div>
      </header>

      <main className={styles.main}>
        {apiError && (
          <div className={styles.errorBanner}>
            Backend no disponible: {apiError}. El predictor usa datos locales; las predicciones requieren el modelo entrenado.
          </div>
        )}

        {activeTab === 'predictor' && (
          <MatchPredictor teams={availableTeams} apiBase={API_BASE} />
        )}
        {activeTab === 'groups' && (
          <GroupTable apiBase={API_BASE} />
        )}
      </main>

      <footer className={styles.footer}>
        Modelo de Poisson jerárquico bayesiano · Mundial 2026
      </footer>
    </div>
  )
}
