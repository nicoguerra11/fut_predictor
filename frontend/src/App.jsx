import { useState, useEffect, useCallback } from 'react'
import MatchPredictor from './components/MatchPredictor'
import GroupTable from './components/GroupTable'
import { TEAMS } from './data/teams'
import styles from './styles/App.module.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [activeTab, setActiveTab] = useState('predictor')
  const [availableTeams, setAvailableTeams] = useState(TEAMS)
  const [apiError, setApiError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)   // { last_trained, teams_in_model }
  const [retraining, setRetraining] = useState(false)
  const [retrainMsg, setRetrainMsg] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/teams`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(() => {
        setAvailableTeams(TEAMS)
      })
      .catch((err) => {
        setApiError(err.message)
      })

    // Cargar info del modelo (cuándo fue entrenado)
    fetch(`${API_BASE}/admin/retrain/status`)
      .then((r) => r.json())
      .then((d) => setModelInfo(d))
      .catch(() => {})
  }, [])

  const handleRetrain = useCallback(() => {
    setRetraining(true)
    setRetrainMsg(null)
    fetch(`${API_BASE}/admin/retrain`, { method: 'POST' })
      .then((r) => r.json())
      .then((d) => {
        setRetrainMsg(d.message || 'Reentrenando...')
        // Polling hasta que termine (~15 segundos)
        const poll = setInterval(() => {
          fetch(`${API_BASE}/admin/retrain/status`)
            .then((r) => r.json())
            .then((s) => {
              setModelInfo(s)
              if (!s.retrain_running) {
                clearInterval(poll)
                setRetraining(false)
                setRetrainMsg('Modelo actualizado con los últimos resultados del Mundial.')
              }
            })
            .catch(() => {
              clearInterval(poll)
              setRetraining(false)
            })
        }, 3000)
      })
      .catch(() => {
        setRetraining(false)
        setRetrainMsg('Error al conectar con el backend.')
      })
  }, [])

  const lastTrained =
    modelInfo?.last_trained === 'build'
      ? 'al deployar'
      : modelInfo?.last_trained
        ? new Date(modelInfo.last_trained).toLocaleString('es-AR')
        : null

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
            Backend no disponible: {apiError}. Las predicciones requieren el modelo entrenado.
          </div>
        )}

        {/* Barra de actualización del modelo */}
        {!apiError && (
          <div className={styles.retrainBar}>
            <div className={styles.retrainInfo}>
              {lastTrained && (
                <span className={styles.retrainTimestamp}>
                  Modelo entrenado {lastTrained}
                </span>
              )}
              {retrainMsg && (
                <span className={styles.retrainMsg}>{retrainMsg}</span>
              )}
            </div>
            <button
              className={styles.retrainBtn}
              onClick={handleRetrain}
              disabled={retraining}
            >
              {retraining ? 'Actualizando...' : 'Actualizar con resultados del Mundial'}
            </button>
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
