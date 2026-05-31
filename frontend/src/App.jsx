import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";
const DEMO_WARNING = "Simulated match for demo only, not real face recognition.";
const PIPELINE_TEXT = "MP4 -> I3D features -> AnomalyNet -> BiLSTM localization -> clip and key frame";

function absoluteMediaUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_BASE_URL}${url}`;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(3);
}

function formatConfidencePercent(confidence) {
  if (confidence === null || confidence === undefined || Number.isNaN(Number(confidence))) {
    return "N/A";
  }
  return `${Number(confidence * 100).toFixed(1)}%`;
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return `${Number(value).toFixed(2)}s`;
}

function formatDate(value) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function statusText(alert) {
  if (alert === true) return "Suspicious";
  if (alert === false) return "Normal";
  return "Failed";
}

function routeFromPath(pathname = window.location.pathname) {
  const match = pathname.match(/^\/results\/([^/]+)$/);
  if (match) return { name: "results", jobId: decodeURIComponent(match[1]) };
  return { name: "home", jobId: null };
}

function filenameFromUrl(url) {
  if (!url) return "";
  const cleanUrl = url.split("?")[0];
  const name = cleanUrl.substring(cleanUrl.lastIndexOf("/") + 1);
  return decodeURIComponent(name);
}

function prettifyVideoName(rawName) {
  if (!rawName) return "";
  let name = rawName;
  const suffixes = [
    "_i3d_features_result.json",
    "_features_result.json",
    "_suspicious_clip.mp4",
    "_suspicious_key_frame.jpg",
    "_result.json",
  ];

  for (const suffix of suffixes) {
    if (name.endsWith(suffix)) {
      name = `${name.slice(0, -suffix.length)}.mp4`;
      break;
    }
  }

  return name;
}

function getVideoFilename(job) {
  if (!job) return "Selected video";
  const explicitName =
    job.video_filename ||
    job.uploaded_filename ||
    job.original_filename ||
    job.filename ||
    job.input_video_filename;
  if (explicitName) return explicitName;

  const media = job.media_urls || {};
  const derived =
    filenameFromUrl(media.result_json) ||
    filenameFromUrl(media.suspicious_clip) ||
    filenameFromUrl(media.key_frame);

  return prettifyVideoName(derived) || `Job ${job.job_id?.slice(0, 8) || "result"}`;
}

function navigateTo(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function Header({ route, theme, onToggleTheme }) {
  return (
    <header className="navbar">
      <button type="button" className="brand-button" onClick={() => navigateTo("/")}>
        <span className="brand-mark">SD</span>
        <span>
          <strong>Shoplifting Detection System</strong>
        </span>
      </button>

      <nav className="nav-actions">
        {route.name === "results" && (
          <button type="button" className="ghost-button" onClick={() => navigateTo("/")}>
            Back to Home
          </button>
        )}
        <button type="button" className="theme-toggle" onClick={onToggleTheme}>
          <span>{theme === "dark" ? "Dark" : "Light"}</span>
          <strong>{theme === "dark" ? "Switch to Light" : "Switch to Dark"}</strong>
        </button>
      </nav>
    </header>
  );
}

function PipelineStrip() {
  return (
    <div className="pipeline-strip">
      <span>Pipeline</span>
      <strong>{PIPELINE_TEXT}</strong>
    </div>
  );
}

function UploadPanel({ onDetectionComplete }) {
  const [file, setFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Please choose an MP4 video first.");
      return;
    }

    setIsRunning(true);
    setError("");
    setMessage("Uploading video and running the detection pipeline. This can take a few minutes.");

    try {
      const formData = new FormData();
      formData.append("video", file);

      const response = await fetch(`${API_BASE_URL}/api/detect`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        const detail = payload?.detail;
        throw new Error(typeof detail === "string" ? detail : detail?.error || "Detection failed.");
      }

      onDetectionComplete({ ...payload, video_filename: file.name });
    } catch (err) {
      setError(err.message || "Detection failed.");
      setMessage("");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <section className="card upload-card">
      <div className="card-heading">
        <p className="eyebrow">New Analysis</p>
        <h1>Upload retail footage</h1>
        <p className="muted">
          Submit an MP4 clip and review the generated localization artifacts in a clean dashboard.
        </p>
      </div>

      <PipelineStrip />

      <form onSubmit={handleSubmit} className="upload-form">
        <label className={`file-input ${file ? "has-file" : ""}`}>
          <input
            type="file"
            accept="video/mp4"
            disabled={isRunning}
            onChange={(event) => {
              setFile(event.target.files?.[0] || null);
              setError("");
              setMessage("");
            }}
          />
          <span className="upload-glyph">MP4</span>
          <strong>{file ? file.name : "Choose MP4 video"}</strong>
          <small>{file ? "Ready to analyze" : "Drag-style upload area for demo footage"}</small>
        </label>

        <button className="primary-button" type="submit" disabled={isRunning || !file}>
          {isRunning ? "Detection running" : "Detect"}
        </button>
      </form>

      {isRunning && (
        <div className="loading-box">
          <span className="spinner" />
          <div>
            <strong>Processing video</strong>
            <p>{message}</p>
          </div>
        </div>
      )}

      {error && <div className="status-message error">{error}</div>}
    </section>
  );
}

function AlertBadge({ alert }) {
  return <span className={`alert-badge ${alert ? "suspicious" : "normal"}`}>{statusText(alert)}</span>;
}

function EmptyHistory() {
  return (
    <div className="empty-history">
      <span>NO JOBS</span>
      <strong>No detection history yet</strong>
      <p>Upload an MP4 video to create the first demo record.</p>
    </div>
  );
}

function HistorySection({ jobs, isLoading, error, onDeleteJob, onClearJobs, deletingIds }) {
  return (
    <section className="card history-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Recent History</p>
          <h2>Previous jobs</h2>
        </div>
        <div className="history-actions">
          {isLoading && <span className="loading-pill">Loading</span>}
          {jobs.length > 0 && (
            <button type="button" className="link-button" onClick={onClearJobs}>
              Clear demo records
            </button>
          )}
        </div>
      </div>

      {error && <div className="status-message error">{error}</div>}

      <div className="history-list">
        {jobs.length === 0 && !isLoading && <EmptyHistory />}
        {jobs.map((job) => (
          <article className="history-item" key={job.job_id}>
            <button
              type="button"
              className="history-open"
              onClick={() => navigateTo(`/results/${encodeURIComponent(job.job_id)}`)}
            >
              <span>
                <strong>{getVideoFilename(job)}</strong>
                <small>{formatDate(job.created_at)}</small>
              </span>
              <AlertBadge alert={job.alert} />
            </button>
            <button
              type="button"
              className="delete-button"
              disabled={deletingIds.has(job.job_id)}
              onClick={(event) => {
                event.stopPropagation();
                onDeleteJob(job.job_id);
              }}
              aria-label={`Delete ${getVideoFilename(job)}`}
            >
              {deletingIds.has(job.job_id) ? "Deleting" : "Delete"}
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}

function HomePage({
  history,
  isHistoryLoading,
  historyError,
  onDetectionComplete,
  onDeleteJob,
  onClearJobs,
  deletingIds,
}) {
  const completedJobs = useMemo(
    () => history.filter((job) => job.status === "completed" || job.status === "failed"),
    [history]
  );

  return (
    <div className="home-layout">
      <UploadPanel onDetectionComplete={onDetectionComplete} />
      <HistorySection
        jobs={completedJobs}
        isLoading={isHistoryLoading}
        error={historyError}
        onDeleteJob={onDeleteJob}
        onClearJobs={() => onClearJobs(completedJobs)}
        deletingIds={deletingIds}
      />
    </div>
  );
}

function MetricCard({ label, value, tone = "default" }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProfileCard({ simulatedMatch }) {
  const profile = simulatedMatch?.matched_profile;

  return (
    <section className="card profile-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Mock Person Match</p>
          <h2>Simulated matched profile</h2>
        </div>
      </div>

      <div className="warning">{DEMO_WARNING}</div>

      {!profile ? (
        <p className="empty-state">No simulated profile was returned for this job.</p>
      ) : (
        <div className="profile-grid">
          {simulatedMatch?.suspect_crop_url && (
            <img
              className="crop-image"
              src={absoluteMediaUrl(simulatedMatch.suspect_crop_url)}
              alt="Simulated suspect crop"
            />
          )}
          <div className="profile-details">
            <h3>{profile.name}</h3>
            <dl>
              <div>
                <dt>Profile ID</dt>
                <dd>{profile.person_id}</dd>
              </div>
              <div>
                <dt>Age</dt>
                <dd>{profile.age}</dd>
              </div>
              <div>
                <dt>Gender</dt>
                <dd>{profile.gender}</dd>
              </div>
              <div>
                <dt>Last seen store</dt>
                <dd>{profile.last_seen_store}</dd>
              </div>
              <div>
                <dt>Previous incidents</dt>
                <dd>{profile.previous_incidents}</dd>
              </div>
              <div>
                <dt>Risk level</dt>
                <dd className={`risk risk-${profile.risk_level}`}>{profile.risk_level}</dd>
              </div>
              <div>
                <dt>Mock confidence</dt>
                <dd>{formatConfidencePercent(simulatedMatch?.simulated_confidence)}</dd>
              </div>
            </dl>
            <p>{profile.notes}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function TopWindowsTable({ windows }) {
  return (
    <section className="card table-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Temporal Model</p>
          <h2>Top-K suspicious windows</h2>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Segment</th>
              <th>Score</th>
              <th>Start</th>
              <th>End</th>
            </tr>
          </thead>
          <tbody>
            {windows.length === 0 ? (
              <tr>
                <td colSpan="5">No suspicious windows returned.</td>
              </tr>
            ) : (
              windows.map((window, index) => (
                <tr key={`${window.rank || index}-${window.segment_index}`}>
                  <td>{window.rank ?? index + 1}</td>
                  <td>{window.segment_index ?? "N/A"}</td>
                  <td>{formatScore(window.score)}</td>
                  <td>{formatSeconds(window.start_time_sec)}</td>
                  <td>{formatSeconds(window.end_time_sec)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ResultsPage({ route, cachedResult }) {
  const [result, setResult] = useState(cachedResult?.job_id === route.jobId ? cachedResult : null);
  const [isLoading, setIsLoading] = useState(!result);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;

    async function fetchResult() {
      if (cachedResult?.job_id === route.jobId) {
        setResult(cachedResult);
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError("");
      try {
        const response = await fetch(`${API_BASE_URL}/api/results/${route.jobId}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.detail || "Could not load result.");
        if (!ignore) setResult(payload);
      } catch (err) {
        if (!ignore) setError(err.message || "Could not load result.");
      } finally {
        if (!ignore) setIsLoading(false);
      }
    }

    fetchResult();
    return () => {
      ignore = true;
    };
  }, [route.jobId, cachedResult]);

  if (isLoading) {
    return (
      <section className="card centered-state">
        <span className="spinner large" />
        <h1>Loading result</h1>
        <p>Fetching detection artifacts for this job.</p>
      </section>
    );
  }

  if (error || !result) {
    return (
      <section className="card centered-state">
        <h1>Result unavailable</h1>
        <p>{error || "No result was found for this job."}</p>
        <button type="button" className="primary-button compact" onClick={() => navigateTo("/")}>
          Back to Home
        </button>
      </section>
    );
  }

  const timestamp = result.suspicious_timestamp || {};
  const media = result.media_urls || {};
  const windows = result.top_k_windows || [];
  const videoName = getVideoFilename(result);

  return (
    <div className="results-page">
      <section className="card result-hero">
        <div>
          <p className="eyebrow">Detection Result</p>
          <h1>{videoName}</h1>
          <p className="muted">Job ID: {result.job_id}</p>
        </div>
        <AlertBadge alert={result.alert} />
      </section>

      {result.error ? (
        <section className="card">
          <div className="status-message error">{result.error}</div>
        </section>
      ) : (
        <>
          <section className="summary-grid">
            <MetricCard label="Alert status" value={statusText(result.alert)} tone={result.alert ? "alert" : "ok"} />
            <MetricCard label="Stage 1 score" value={formatScore(result.stage1_score)} />
            <MetricCard
              label="Suspicious timestamp"
              value={`${formatSeconds(timestamp.start_sec)} - ${formatSeconds(timestamp.end_sec)}`}
            />
          </section>

          <section className="card media-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Evidence Artifacts</p>
                <h2>Suspicious clip and key frame</h2>
              </div>
            </div>
            <div className="media-grid">
              <div className="media-block">
                <h3>Suspicious clip</h3>
                {media.suspicious_clip ? (
                  <video controls src={absoluteMediaUrl(media.suspicious_clip)} />
                ) : (
                  <p className="empty-state">No suspicious clip returned.</p>
                )}
              </div>
              <div className="media-block">
                <h3>Key frame</h3>
                {media.key_frame ? (
                  <img src={absoluteMediaUrl(media.key_frame)} alt="Suspicious key frame" />
                ) : (
                  <p className="empty-state">No key frame returned.</p>
                )}
              </div>
            </div>
          </section>

          <TopWindowsTable windows={windows} />
          <ProfileCard simulatedMatch={result.simulated_match} />
        </>
      )}
    </div>
  );
}

function getInitialTheme() {
  const savedTheme = localStorage.getItem("shoplifting-dashboard-theme");
  if (savedTheme === "light" || savedTheme === "dark") return savedTheme;
  return "light";
}

export default function App() {
  const [route, setRoute] = useState(routeFromPath);
  const [theme, setTheme] = useState(getInitialTheme);
  const [latestResult, setLatestResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [deletingIds, setDeletingIds] = useState(new Set());

  async function fetchHistory() {
    setIsHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch(`${API_BASE_URL}/api/history`);
      const payload = await response.json();
      if (!response.ok) throw new Error("Could not load job history.");
      setHistory(Array.isArray(payload) ? payload : []);
    } catch (err) {
      setHistoryError(err.message || "Could not load job history.");
    } finally {
      setIsHistoryLoading(false);
    }
  }

  async function deleteJob(jobId, { confirmFirst = true } = {}) {
    if (confirmFirst && !window.confirm("Delete this detection record?")) return false;

    setDeletingIds((current) => new Set(current).add(jobId));
    setHistoryError("");
    try {
      const response = await fetch(`${API_BASE_URL}/api/results/${jobId}`, {
        method: "DELETE",
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || "Could not delete job.");

      setHistory((items) => items.filter((item) => item.job_id !== jobId));
      if (latestResult?.job_id === jobId) setLatestResult(null);
      return true;
    } catch (err) {
      setHistoryError(err.message || "Could not delete job.");
      return false;
    } finally {
      setDeletingIds((current) => {
        const next = new Set(current);
        next.delete(jobId);
        return next;
      });
    }
  }

  async function clearJobs(jobs) {
    if (jobs.length === 0) return;
    if (!window.confirm("Delete all demo detection records?")) return;

    for (const job of jobs) {
      await deleteJob(job.job_id, { confirmFirst: false });
    }
  }

  function handleDetectionComplete(result) {
    setLatestResult(result);
    fetchHistory();
    navigateTo(`/results/${encodeURIComponent(result.job_id)}`);
  }

  function toggleTheme() {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("shoplifting-dashboard-theme", theme);
  }, [theme]);

  useEffect(() => {
    function handlePopState() {
      setRoute(routeFromPath());
    }

    window.addEventListener("popstate", handlePopState);
    fetchHistory();
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  return (
    <main className="app-shell">
      <Header route={route} theme={theme} onToggleTheme={toggleTheme} />

      {route.name === "results" ? (
        <ResultsPage route={route} cachedResult={latestResult} />
      ) : (
        <HomePage
          history={history}
          isHistoryLoading={isHistoryLoading}
          historyError={historyError}
          onDetectionComplete={handleDetectionComplete}
          onDeleteJob={deleteJob}
          onClearJobs={clearJobs}
          deletingIds={deletingIds}
        />
      )}
    </main>
  );
}
