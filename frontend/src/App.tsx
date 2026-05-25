import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import brandMark from './assets/brand-mark.svg';
import {
  MapPin,
  Clock,
  ExternalLink,
  Search,
  CheckCircle,
  Loader2,
  Play,
  Upload,
  SkipForward,
  FileText,
  Sparkles,
  Briefcase,
  GraduationCap,
  Award,
  Code2,
  Send
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

interface Job {
  id: string;
  source: string;
  job_title: string;
  company: string;
  location: string;
  remote: boolean;
  posted_date: string;
  url: string;
  description: string;
  status: string;
  salary?: string;
}

interface Summary {
  total_discovered: number;
  total_deduplicated: number;
  by_source: Record<string, number>;
  last_scan_time?: string;
  groq_api_usage?: number;
  tailored?: number;
  ready?: number;
  submitted?: number;
}

interface AppStatus {
  running: boolean;
  current_job?: { title: string; company: string; id?: string };
  current_idx?: number;
  total?: number;
  error?: string;
}

interface DiscoveryLogs {
  lines: string[];
}

interface AppliedJob {
  job_id: string;
  job_title: string;
  company: string;
  status: string;
  submitted_at?: string;
  job_url?: string;
  source?: string;
  location?: string;
  posted_date?: string;
  tailored_cv?: any;
  tailored_cv_latex?: string;
  cover_letter?: string;
}

type Tab = 'overview' | 'start';
type AutoApplyPhase = 'idle' | 'await_cv' | 'tailoring' | 'countdown' | 'starting_bot';

const App: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [allJobs, setAllJobs] = useState<Job[]>([]);
  const [appliedJobs, setAppliedJobs] = useState<AppliedJob[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [appStatus, setAppStatus] = useState<AppStatus>({ running: false });
  const [discoveryRunning, setDiscoveryRunning] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [profile, setProfile] = useState<any>(null);
  const [tailoredPreview, setTailoredPreview] = useState<any>(null);
  const [discoveryLogs, setDiscoveryLogs] = useState<string[]>([]);
  const [discoveryMessage, setDiscoveryMessage] = useState<string>('');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [overviewMode, setOverviewMode] = useState<'all' | 'unique' | 'pipeline'>('unique');
  const [waitingForReviewCompletion, setWaitingForReviewCompletion] = useState(false);
  const [expandedAppliedJobId, setExpandedAppliedJobId] = useState<string | null>(null);
  const [cvVersion, setCvVersion] = useState(0);
  const [showBlueAlert, setShowBlueAlert] = useState<string | null>(null);
  const [autoApplyChain, setAutoApplyChain] = useState<{
    active: boolean;
    jobId: string | null;
    phase: AutoApplyPhase;
    countdown: number;
    error?: string;
  }>({ active: false, jobId: null, phase: 'idle', countdown: 0 });

  const autoApplyTimerRef = React.useRef<number | null>(null);
  const autoApplyStateRef = React.useRef(autoApplyChain);

  useEffect(() => {
    autoApplyStateRef.current = autoApplyChain;
  }, [autoApplyChain]);

  const clearAutoApplyTimer = () => {
    if (autoApplyTimerRef.current) {
      window.clearInterval(autoApplyTimerRef.current);
      autoApplyTimerRef.current = null;
    }
  };

  const cancelAutoApplyChain = (reason?: string) => {
    clearAutoApplyTimer();
    setAutoApplyChain({ active: false, jobId: null, phase: 'idle', countdown: 0, error: reason });
    if (reason) {
      setShowBlueAlert(reason);
      setTimeout(() => setShowBlueAlert(null), 4000);
    }
  };

  const hasProfile = useMemo(() => {
    return Boolean(profile && typeof profile === 'object' && Object.keys(profile).length > 0);
  }, [profile]);

  const formatPST = (isoString?: string) => {
    if (!isoString) return 'Never';
    return new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Karachi',
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(isoString)) + ' PST';
  };

  const fetchData = async () => {
    try {
      const safeGet = async <T,>(url: string, fallback: T): Promise<T> => {
        try {
          const res = await axios.get<T>(url);
          return res.data;
        } catch {
          return fallback;
        }
      };

      const [uniqueJobs, allJobsData, profileData, discoveryStatusData, appStatusData, summaryData, appliedJobsData, logsData] =
        await Promise.all([
          safeGet<Job[]>(`${API_BASE}/jobs?mode=unique&t=${Date.now()}`, []),
          safeGet<Job[]>(`${API_BASE}/jobs?mode=all&t=${Date.now()}`, []),
          safeGet<any>(`${API_BASE}/cv/profile?t=${Date.now()}`, null),
          safeGet<any>(`${API_BASE}/discover/status?t=${Date.now()}`, { running: false, last_message: '' }),
          safeGet<AppStatus>(`${API_BASE}/apply/status?t=${Date.now()}`, { running: false }),
          safeGet<Summary | null>(`${API_BASE}/summary?t=${Date.now()}`, null),
          safeGet<AppliedJob[]>(`${API_BASE}/jobs/applied?t=${Date.now()}`, []),
          safeGet<DiscoveryLogs>(`${API_BASE}/discover/logs?limit=18&t=${Date.now()}`, { lines: [] })
        ]);

      setJobs(uniqueJobs);
      setAllJobs(allJobsData);
      setProfile(profileData);
      setDiscoveryRunning(Boolean(discoveryStatusData?.running));
      setDiscoveryMessage(discoveryStatusData?.last_message || '');
      setDiscoveryLogs(logsData.lines || []);
      setAppStatus(appStatusData);
      setSummary(summaryData);
      setAppliedJobs(appliedJobsData || []);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (appStatus.running && appStatus.current_job?.id) {
      setSelectedJobId(appStatus.current_job.id);
      setActiveTab('start');
    }
  }, [appStatus.running, appStatus.current_job?.id]);

  useEffect(() => {
    if (waitingForReviewCompletion && !appStatus.running) {
      setWaitingForReviewCompletion(false);
      setSelectedJobId(null);
      setTailoredPreview(null);
      setActiveTab('overview');
      fetchData();
    }
  }, [waitingForReviewCompletion, appStatus.running]);

  useEffect(() => {
    const loadTailoredForSelected = async () => {
      if (!selectedJobId) {
        return;
      }
      try {
        const tailoredRes = await axios.get(`${API_BASE}/jobs/tailored/${selectedJobId}`);
        setTailoredPreview(tailoredRes.data);
      } catch {
        setTailoredPreview(null);
      }
    };
    loadTailoredForSelected();
  }, [selectedJobId]);

  const selectedJob = useMemo(() => {
    if (!selectedJobId) {
      return null;
    }
    return allJobs.find((j) => String(j.id) === String(selectedJobId)) ||
      jobs.find((j) => String(j.id) === String(selectedJobId)) ||
      null;
  }, [selectedJobId, allJobs, jobs]);

  const overviewJobs = useMemo(() => {
    let filtered = (overviewMode === 'all' ? allJobs : jobs);
    
    if (overviewMode === 'pipeline') {
      // In pipeline mode, show jobs that are tailored or ready but not yet submitted
      filtered = allJobs.filter(j => j.status === 'tailored' || j.status === 'ready');
    }

    return filtered.filter((job) =>
      job.job_title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.company.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [overviewMode, allJobs, jobs, searchTerm]);

  const readyToApplyCount = Math.max((summary?.total_discovered || 0) - (summary?.total_deduplicated || 0), 0);
  const appliedCount = appliedJobs.length;

  const handleStartDiscovery = async () => {
    setDiscoveryRunning(true);
    try {
      await axios.post(`${API_BASE}/discover`);
      await fetchData();
    } catch (error) {
      console.error(error);
    }
  };

  const handleSelectJobToApply = async (jobId: string) => {
    // Manual mode: just select the job and take user to the Start tab.
    // Do not clear CV automatically (Auto Apply chain depends on reusing the uploaded CV).
    setSelectedJobId(jobId);
    setTailoredPreview(null);
    setActiveTab('start');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    await fetchData();
  };

  const tailorSelectedJob = async (jobId: string) => {
    setIsGeneratingPreview(true);
    try {
      const tailorRes = await axios.post(`${API_BASE}/jobs/tailor/${jobId}`);
      if (tailorRes.data?.tailored) {
        setTailoredPreview(tailorRes.data.tailored);
        return tailorRes.data.tailored;
      }
      const fallback = await axios.get(`${API_BASE}/jobs/tailored/${jobId}`);
      setTailoredPreview(fallback.data);
      return fallback.data;
    } finally {
      setIsGeneratingPreview(false);
      await fetchData();
    }
  };

  const startAutoApplyChain = async (jobId: string) => {
    clearAutoApplyTimer();
    setSelectedJobId(jobId);
    setTailoredPreview(null);
    setActiveTab('start');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    setAutoApplyChain({ active: true, jobId, phase: 'await_cv', countdown: 0 });

    try {
      const profileRes = await axios.get(`${API_BASE}/cv/profile?t=${Date.now()}`);
      const nextProfile = profileRes.data;
      setProfile(nextProfile);

      const hasCv = Boolean(nextProfile && typeof nextProfile === 'object' && Object.keys(nextProfile).length > 0);
      if (!hasCv) {
        setShowBlueAlert('Auto Apply: upload your CV to continue.');
        setTimeout(() => setShowBlueAlert(null), 4500);
        setAutoApplyChain({ active: true, jobId, phase: 'await_cv', countdown: 0 });
        return;
      }

      setAutoApplyChain({ active: true, jobId, phase: 'tailoring', countdown: 0 });
      await tailorSelectedJob(jobId);

      // Safety window: allow cancellation before launching the bot.
      setAutoApplyChain({ active: true, jobId, phase: 'countdown', countdown: 3 });
      clearAutoApplyTimer();
      autoApplyTimerRef.current = window.setInterval(() => {
        setAutoApplyChain((prev) => {
          if (!prev.active || prev.jobId !== jobId || prev.phase !== 'countdown') {
            return prev;
          }
          const next = Math.max(0, (prev.countdown || 0) - 1);
          return { ...prev, countdown: next };
        });
      }, 1000);

      window.setTimeout(async () => {
        clearAutoApplyTimer();
        // Re-check current chain state before starting apply.
        const current = autoApplyStateRef.current;
        if (!current.active || current.jobId !== jobId) return;
        if (current.phase !== 'countdown') return;

        try {
          setAutoApplyChain({ active: true, jobId, phase: 'starting_bot', countdown: 0 });
          await axios.post(`${API_BASE}/apply/start`, [jobId]);
          await fetchData();
          cancelAutoApplyChain();
        } catch (e: any) {
          const detail = e.response?.data?.detail || e.message || 'Unknown error';
          cancelAutoApplyChain(`Auto Apply failed to start bot: ${detail}`);
        }
      }, 3100);
    } catch (e: any) {
      const detail = e.response?.data?.detail || e.message || 'Unknown error';
      cancelAutoApplyChain(`Auto Apply failed: ${detail}`);
    }
  };

  const normalizeSkills = (skillsData: any): string[] => {
    if (!skillsData) return [];
    if (Array.isArray(skillsData)) {
      return skillsData.map(s => {
        if (typeof s === 'string') return s;
        if (typeof s === 'object' && s !== null) {
          if (s.technologies && Array.isArray(s.technologies)) {
            return `${s.category}: ${s.technologies.join(', ')}`;
          }
          return JSON.stringify(s);
        }
        return String(s);
      });
    }
    if (typeof skillsData === 'object' && skillsData !== null) {
      return Object.entries(skillsData).map(([key, val]) => {
        if (Array.isArray(val)) return `${key}: ${val.join(', ')}`;
        return `${key}: ${val}`;
      });
    }
    return [];
  };

  const normalizeLinks = (linksData: any): [string, string][] => {
    if (!linksData) return [];
    if (typeof linksData === 'object' && !Array.isArray(linksData)) {
      return Object.entries(linksData)
        .filter(([_, url]) => url && typeof url === 'string')
        .map(([name, url]) => [name, url as string]);
    }
    if (Array.isArray(linksData)) {
      return linksData.map(l => {
        if (Array.isArray(l) && l.length === 2) return [String(l[0]), String(l[1])];
        if (typeof l === 'string') return [l, l];
        return ['Link', JSON.stringify(l)];
      });
    }
    return [];
  };

  const handleGenerateTailoredDocs = async () => {
    if (!selectedJobId) {
      return;
    }
    if (!hasProfile) {
      alert('Please upload CV first.');
      return;
    }
    try {
      await tailorSelectedJob(selectedJobId);
    } catch (error) {
      console.error(error);
      alert('Failed to generate tailored CV and cover letter.');
    }
  };

  const handleOpenApplyForm = async () => {
    if (!selectedJobId) {
      return;
    }
    if (!hasProfile) {
      setShowBlueAlert('Please upload CV first.');
      setTimeout(() => setShowBlueAlert(null), 4000);
      return;
    }
    if (!tailoredPreview) {
      setShowBlueAlert('Please generate tailored CV and cover letter first.');
      setTimeout(() => setShowBlueAlert(null), 4000);
      return;
    }
    try {
      await axios.post(`${API_BASE}/apply/start`, [selectedJobId]);
      await fetchData();
    } catch (e: any) {
      console.error(e);
      const detail = e.response?.data?.detail || e.message || 'Unknown error';
      alert(`Failed to start apply form. Error: ${detail}`);
    }
  };

  const handleReviewComplete = async () => {
    try {
      setWaitingForReviewCompletion(true);
      await axios.post(`${API_BASE}/apply/review/complete`);
    } catch (error) {
      console.error(error);
      setWaitingForReviewCompletion(false);
    }
  };

  const handleSkipApplication = async () => {
    try {
      setWaitingForReviewCompletion(true);
      await axios.post(`${API_BASE}/apply/skip`);
    } catch (error) {
      console.error(error);
      setWaitingForReviewCompletion(false);
    }
  };

  const [isUploading, setIsUploading] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const selectedFile = e.target.files[0];
    const formData = new FormData();
    formData.append('file', selectedFile);
    setIsUploading(true);
    try {
      const res = await axios.post(`${API_BASE}/cv/upload`, formData);
      // Immediately update profile from the response so the UI reflects the new CV
      if (res.data?.profile) {
        setProfile(res.data.profile);
      }
      setTailoredPreview(null);
      setCvVersion((v) => v + 1);
      // Also refresh all data to keep everything in sync
      await fetchData();

      // If Auto Apply chain is active, continue immediately with tailoring.
      if (autoApplyChain.active && autoApplyChain.jobId) {
        const chainJobId = autoApplyChain.jobId;
        setSelectedJobId(chainJobId);
        setActiveTab('start');
        setAutoApplyChain({ active: true, jobId: chainJobId, phase: 'tailoring', countdown: 0 });
        try {
          await tailorSelectedJob(chainJobId);
          setAutoApplyChain({ active: true, jobId: chainJobId, phase: 'countdown', countdown: 3 });
          clearAutoApplyTimer();
          autoApplyTimerRef.current = window.setInterval(() => {
            setAutoApplyChain((prev) => {
              if (!prev.active || prev.jobId !== chainJobId || prev.phase !== 'countdown') return prev;
              const next = Math.max(0, (prev.countdown || 0) - 1);
              return { ...prev, countdown: next };
            });
          }, 1000);
          window.setTimeout(async () => {
            clearAutoApplyTimer();
            const current = autoApplyStateRef.current;
            if (!current.active || current.jobId !== chainJobId) return;
            if (current.phase !== 'countdown') return;
            try {
              setAutoApplyChain({ active: true, jobId: chainJobId, phase: 'starting_bot', countdown: 0 });
              await axios.post(`${API_BASE}/apply/start`, [chainJobId]);
              await fetchData();
              cancelAutoApplyChain();
            } catch (err: any) {
              const detail = err.response?.data?.detail || err.message || 'Unknown error';
              cancelAutoApplyChain(`Auto Apply failed to start bot: ${detail}`);
            }
          }, 3100);
        } catch (err: any) {
          const detail = err.response?.data?.detail || err.message || 'Unknown error';
          cancelAutoApplyChain(`Auto Apply tailoring failed: ${detail}`);
        }
      }
    } catch {
      alert('Failed to upload CV');
    } finally {
      setIsUploading(false);
      // Allow selecting the same file again and still trigger onChange.
      e.target.value = '';
    }
  };

  const normalizeBullets = (desc: any): string[] => {
    if (Array.isArray(desc)) return desc.map(String).filter(s => s.trim());
    if (typeof desc === 'string') return desc.split('\n').map(s => s.replace(/^[-•*]\s*/, '').trim()).filter(Boolean);
    return [];
  };

  const renderTailoredResume = (cv: any) => {
    if (!cv || typeof cv !== 'object') return null;
    const skills = normalizeSkills(cv.skills);
    const experience: any[] = Array.isArray(cv.experience) ? cv.experience : [];
    const education: any[] = Array.isArray(cv.education) ? cv.education : [];
    const projects: any[] = Array.isArray(cv.projects) ? cv.projects : [];
    const certifications: any[] = Array.isArray(cv.certifications) ? cv.certifications : typeof cv.certifications === 'string' ? [cv.certifications] : [];
    const links = normalizeLinks(cv.links);

    return (
      <div className="tailored-resume">
        <div className="resume-header-block">
          <h3 className="resume-name">{String(cv.full_name || 'Candidate')}</h3>
          <div className="resume-contact-row">
            {cv.email && <span>✉ {String(cv.email)}</span>}
            {cv.phone && <span>☎ {String(cv.phone)}</span>}
            {cv.location && <span><MapPin size={13} /> {String(cv.location)}</span>}
          </div>
          {links.length > 0 && (
            <div className="resume-contact-row">
              {links.map(([name, url], i) => (
                <a key={i} href={url.startsWith('http') ? url : `https://${url}`} target="_blank" rel="noopener noreferrer" className="resume-link">
                  {name.toLowerCase().includes('linkedin') ? '🔗 LinkedIn' : name.toLowerCase().includes('github') ? '🔗 GitHub' : `🔗 ${name}`}
                </a>
              ))}
            </div>
          )}
        </div>

        {cv.summary && (
          <div className="resume-section-block">
            <h4><Briefcase size={15} /> Professional Summary</h4>
            <p>{String(cv.summary)}</p>
          </div>
        )}

        {skills.length > 0 && (
          <div className="resume-section-block">
            <h4><Code2 size={15} /> Technical Skills</h4>
            <div className="tag-cloud">{skills.map((s: string, i: number) => <span key={i} className="badge">{s}</span>)}</div>
          </div>
        )}

        {experience.length > 0 && (
          <div className="resume-section-block">
            <h4><Briefcase size={15} /> Experience</h4>
            {experience.map((exp: any, i: number) => {
              const bullets = normalizeBullets(exp.description || exp.bullet_points || exp.highlights);
              return (
                <div key={i} className="resume-entry">
                  <div className="entry-top-row">
                    <strong>{String(exp.role || exp.title || exp.job_title || exp.position || 'Role')}</strong>
                    <span className="entry-duration">{String(exp.duration || exp.dates || exp.period || '')}</span>
                  </div>
                  <div className="entry-org">{String(exp.company || exp.organization || '')}</div>
                  {bullets.length > 0 && <ul className="entry-bullets">{bullets.map((b: string, j: number) => <li key={j}>{b}</li>)}</ul>}
                </div>
              );
            })}
          </div>
        )}

        {education.length > 0 && (
          <div className="resume-section-block">
            <h4><GraduationCap size={15} /> Education</h4>
            {education.map((edu: any, i: number) => (
              <div key={i} className="resume-entry">
                <div className="entry-top-row">
                  <strong>{String(edu.degree || edu.title || edu.program || 'Degree')}</strong>
                  <span className="entry-duration">{String(edu.year || edu.duration || edu.dates || '')}</span>
                </div>
                <div className="entry-org">{String(edu.institution || edu.school || edu.university || '')}</div>
              </div>
            ))}
          </div>
        )}

        {projects.length > 0 && (
          <div className="resume-section-block">
            <h4><Code2 size={15} /> Projects</h4>
            {projects.map((proj: any, i: number) => {
              const bullets = normalizeBullets(proj.description);
              return (
                <div key={i} className="resume-entry">
                  <strong>{String(proj.title || proj.name || 'Project')}</strong>
                  {bullets.length > 0 && <ul className="entry-bullets">{bullets.map((b: string, j: number) => <li key={j}>{b}</li>)}</ul>}
                </div>
              );
            })}
          </div>
        )}

        {certifications.length > 0 && (
          <div className="resume-section-block">
            <h4><Award size={15} /> Certifications</h4>
            <ul className="entry-bullets">{certifications.map((c: any, i: number) => <li key={i}>{String(c)}</li>)}</ul>
          </div>
        )}
      </div>
    );
  };

  const renderJobCard = (job: Job, index: number) => (
    <motion.div
      key={job.id}
      className={`job-card-v2 status-${job.status}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.03 }}
    >
      <div className="card-top">
        <span className="source-tag">{job.source}</span>
        <span className={`status-tag ${job.status}`}>{job.status}</span>
      </div>
      <div className="card-body">
        <h4 className="job-title">{job.job_title}</h4>
        <p className="company-name">{job.company}</p>
        <div className="meta-info">
          <span><MapPin size={14} /> {job.location}</span>
          <span><Clock size={14} /> {job.posted_date}</span>
        </div>
      </div>
      <div className="card-footer">
        <div className="flex gap-2">
          <a href={job.url} target="_blank" rel="noopener noreferrer" className="icon-btn"><ExternalLink size={18} /></a>
        </div>
        <button
          className="btn btn-primary btn-small"
          onClick={() => startAutoApplyChain(job.id)}
          disabled={appStatus.running}
        >
          <Play size={14} /> Auto Apply
        </button>
      </div>
    </motion.div>
  );

  return (
    <div className="app-container">
      <header className="main-header">
        <div className="logo-section">
          <motion.div className="logo-icon" animate={{ rotate: 360 }} transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}>
            <img src={brandMark} alt="" aria-hidden="true" />
          </motion.div>
          <div>
            <h1>Job Bot <span className="text-gradient">Pro</span></h1>
            <p>Automated Data Engineering Career Agent</p>
          </div>
          {summary?.groq_api_usage !== undefined && (
            <div className="api-usage-badge ml-4">
              <Sparkles size={14} />
              <span>Groq Calls: {summary.groq_api_usage}</span>
            </div>
          )}
        </div>

        <div className="header-actions">
          <div className="search-bar">
            <Search size={18} />
            <input type="text" placeholder="Filter roles..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />
          </div>
          <button className={`btn btn-primary ${discoveryRunning ? 'loading' : ''}`} onClick={handleStartDiscovery} disabled={discoveryRunning}>
            {discoveryRunning ? <Loader2 className="spin" /> : <Play size={18} />}
            {discoveryRunning ? 'Scanning Jobs' : 'Scan Jobs'}
          </button>
        </div>
      </header>

      {(discoveryRunning || discoveryLogs.length > 0) && (
        <section className={`scan-status-panel ${discoveryRunning ? 'active' : ''}`}>
          <div className="scan-status-header">
            <div>
              <h3>{discoveryRunning ? 'Discovery running' : 'Last discovery run'}</h3>
              <p>{discoveryMessage || (discoveryRunning ? 'Fetching jobs from Apify actors.' : `Last successful run: ${formatPST(summary?.last_scan_time)}`)}</p>
            </div>
            <span className={`scan-status-pill ${discoveryRunning ? 'live' : 'idle'}`}>
              {discoveryRunning ? 'Live' : 'Idle'}
            </span>
          </div>
          {discoveryLogs.length > 0 && (
            <div className="scan-log-window">
              {discoveryLogs.map((line, index) => (
                <div key={`${index}-${line}`} className="scan-log-line">{line}</div>
              ))}
            </div>
          )}
        </section>
      )}

      <nav className="tab-navigation">
        <button className={`tab-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>Overview</button>
        <button className={`tab-item ${activeTab === 'start' ? 'active' : ''}`} onClick={() => setActiveTab('start')}>Start Application</button>
      </nav>

      <div className="app-content-wrapper">
        {activeTab === 'overview' && (
          <div className="tab-pane overview-pane">
            <h2 className="mb-4">System Overview</h2>
            <div className="metrics-grid">
              <div className={`stat-card cursor-pointer ${overviewMode === 'all' ? 'active-all' : ''}`} onClick={() => setOverviewMode('all')}>
                <div className="stat-label">Total Discovered</div>
                <div className="stat-value">{summary?.total_discovered || 0}</div>
                <div className="source-breakdown mt-2 text-sm text-muted">
                  <div className="flex justify-between"><span>LinkedIn:</span> <span>{summary?.by_source?.LinkedIn || summary?.by_source?.linkedin || 0}</span></div>
                  <div className="flex justify-between"><span>Indeed:</span> <span>{summary?.by_source?.Indeed || summary?.by_source?.indeed || 0}</span></div>
                  <div className="flex justify-between"><span>Dice:</span> <span>{summary?.by_source?.Dice || summary?.by_source?.dice || 0}</span></div>
                </div>
              </div>

              <div className={`stat-card border-success-subtle cursor-pointer ${overviewMode === 'unique' ? 'active-unique' : ''}`} onClick={() => setOverviewMode('unique')}>
                <div className="stat-label">Unique Jobs</div>
                <div className="stat-value text-success">{summary?.total_deduplicated || 0}</div>
                <div className="source-breakdown mt-2 text-sm text-muted">
                  <p>Duplicates automatically removed</p>
                </div>
              </div>

              <div className={`stat-card border-warning-subtle cursor-pointer ${overviewMode === 'pipeline' ? 'active-pipeline' : ''}`} onClick={() => setOverviewMode('pipeline')}>
                <div className="stat-label">Application Pipeline</div>
                <div className="stat-value text-warning">{summary?.tailored || 0}</div>
                <div className="source-breakdown mt-2 text-sm text-muted">
                  <p>Tailored & Ready: {summary?.tailored || 0}</p>
                  <p>Successfully Applied: {appliedCount}</p>
                </div>
              </div>
            </div>

            <div className="overview-jobs-section">
              <div className="overview-jobs-header">
                <div>
                  <h3>
                    {overviewMode === 'all' ? 'All Discovered Jobs' : 
                     overviewMode === 'unique' ? 'Unique Jobs' : 'Tailored Pipeline'}
                  </h3>
                  <p>
                    {overviewMode === 'all'
                      ? 'Showing every discovered role from the latest scan.'
                      : overviewMode === 'unique'
                      ? 'Showing the deduplicated set after duplicate removal.'
                      : 'Showing jobs that are tailored and ready for application.'}
                  </p>
                </div>
                <span className="badge badge-warning">{overviewJobs.length} Jobs</span>
              </div>

              <div className="job-cards-grid overview-job-grid">
                <AnimatePresence>
                  {overviewJobs.filter((j) => j.status !== 'submitted').map((job, index) => renderJobCard(job, index))}
                </AnimatePresence>
              </div>
            </div>

            <div className="overview-jobs-section">
              <div className="list-header">
                <h3>Applied Jobs</h3>
                <span className="badge badge-warning">{appliedJobs.length} Applied</span>
              </div>
              {appliedJobs.length === 0 && (
                <div className="empty-state">
                  <p>No submitted jobs yet.</p>
                </div>
              )}
              <div className="applied-jobs-list">
                {appliedJobs.map((job) => {
                  const expanded = expandedAppliedJobId === job.job_id;
                  return (
                    <div className="applied-job-card" key={job.job_id}>
                      <div className="applied-job-top">
                        <div>
                          <h4 className="job-title">{job.job_title}</h4>
                          <p className="company-name">{job.company}</p>
                          <div className="meta-info">
                            {job.source && <span>{job.source}</span>}
                            {job.location && <span><MapPin size={14} /> {job.location}</span>}
                            {job.submitted_at && <span><CheckCircle size={14} /> Submitted: {job.submitted_at}</span>}
                          </div>
                        </div>
                        <button className="btn-small" onClick={() => setExpandedAppliedJobId(expanded ? null : job.job_id)}>
                          {expanded ? 'Hide Details' : 'Show Details'}
                        </button>
                      </div>
                      {expanded && (
                        <div className="applied-job-details">
                          <div className="preview-grid">
                            <div className="preview-pane">
                              <div className="pane-header"><FileText size={16} /> Submitted CV (LaTeX)</div>
                              <div className="cover-letter-content">
                                <pre>{job.tailored_cv_latex || 'No LaTeX CV snapshot saved for this job.'}</pre>
                              </div>
                            </div>
                            <div className="preview-pane">
                              <div className="pane-header"><FileText size={16} /> Submitted Cover Letter</div>
                              <div className="cover-letter-content">
                                <pre>{job.cover_letter || 'No cover letter snapshot saved for this job.'}</pre>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'start' && (
          <div className="tab-pane cv-pane">
            {/* No job selected */}
            {!selectedJob && (
              <div className="empty-state">
                <Briefcase size={48} style={{ marginBottom: '1rem', opacity: 0.4 }} />
                <h3>No Job Selected</h3>
                <p>Go to <strong>Overview</strong> and click <strong>Auto Apply</strong> on a job to begin.</p>
              </div>
            )}

            {/* Selected job card */}
            {selectedJob && (
              <>
                <div className="stat-card mb-4" style={{ borderLeft: '4px solid var(--primary)' }}>
                  <div className="list-header">
                    <div>
                      <h3 style={{ fontSize: '1.15rem' }}>Selected Job</h3>
                      <p style={{ fontSize: '1.05rem', fontWeight: 600, marginTop: '0.25rem' }}>{String(selectedJob.job_title)} at {String(selectedJob.company)}</p>
                    </div>
                    <a href={selectedJob.url} target="_blank" rel="noopener noreferrer" className="btn-small">
                      <ExternalLink size={14} /> View Job
                    </a>
                  </div>
                  <div className="meta-info mt-2">
                    <span><MapPin size={14} /> {selectedJob.location}</span>
                    <span><Clock size={14} /> {selectedJob.posted_date}</span>
                  </div>
                </div>

                {/* Blue Alert Popup */}
                {showBlueAlert && (
                  <motion.div 
                    initial={{ opacity: 0, y: -20 }} 
                    animate={{ opacity: 1, y: 0 }}
                    className="alert-banner-blue mb-4"
                  >
                    <Loader2 className="spin" size={18} />
                    {showBlueAlert}
                  </motion.div>
                )}

                {/* Auto Apply chain status */}
                {autoApplyChain.active && autoApplyChain.jobId && String(autoApplyChain.jobId) === String(selectedJobId) && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="stat-card mb-4"
                    style={{ borderLeft: '4px solid var(--primary)' }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <strong>Auto Apply is running</strong>
                        <div className="text-sm text-muted" style={{ marginTop: '0.25rem' }}>
                          {autoApplyChain.phase === 'await_cv' && 'Step 1/3: Upload CV to continue.'}
                          {autoApplyChain.phase === 'tailoring' && 'Step 2/3: Tailoring CV & cover letter…'}
                          {autoApplyChain.phase === 'countdown' && `Step 3/3: Starting bot in ${autoApplyChain.countdown}s (click Cancel to stop).`}
                          {autoApplyChain.phase === 'starting_bot' && 'Launching bot…'}
                        </div>
                      </div>
                      <button className="btn btn-outline" onClick={() => cancelAutoApplyChain('Auto Apply canceled.')} disabled={autoApplyChain.phase === 'starting_bot'}>
                        Cancel
                      </button>
                    </div>
                  </motion.div>
                )}

                {/* Step 1: Upload CV */}
                <div className="upload-section mb-4">
                  <div className="stat-card stat-card-accent text-center">
                    <h4 className="mb-2">Upload CV</h4>
                    <p className="text-sm text-muted mb-4">Use your latest CV before generating job-specific documents.</p>
                    <label className={`btn btn-primary w-full justify-center cursor-pointer ${isUploading ? 'loading' : ''}`} style={{ pointerEvents: isUploading ? 'none' : 'auto', opacity: isUploading ? 0.7 : 1 }}>
                      {isUploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                      {isUploading ? 'Processing CV...' : (profile?.full_name ? 'Update CV' : 'Upload CV')}
                      <input type="file" hidden onChange={handleFileUpload} accept=".pdf,.docx,.doc" disabled={isUploading} />
                    </label>
                  </div>
                </div>

                {/* Step 2: CV Preview + Parsed Profile */}
                {hasProfile && (
                  <div className="parallel-view mb-4">
                    <div className="view-card">
                      <h4>Uploaded CV</h4>
                      <iframe src={`${API_BASE}/cv/file?v=${cvVersion}`} className="cv-iframe" title="Uploaded CV" />
                    </div>
                    <div className="view-card">
                      <h4>Parsed Profile</h4>
                      <div className="parsed-data">
                        <div className="data-group"><label>Full Name</label><div>{String(profile.full_name)}</div></div>
                        <div className="data-group"><label>Summary</label><p>{String(profile.summary)}</p></div>
                        <div className="data-group"><label>Key Skills</label><div className="tag-cloud">{normalizeSkills(profile.skills).map((s: string, i: number) => <span key={i} className="badge">{s}</span>)}</div></div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Step 3: Generate tailored documents */}
                {hasProfile && (
                  <div className="workflow-container">
                    {/* Generate button - only when no preview yet */}
                    {!tailoredPreview && !isGeneratingPreview && (
                      <button className="btn btn-primary generate-btn" onClick={handleGenerateTailoredDocs} disabled={appStatus.running}>
                        <Sparkles size={20} />
                        See Tailored CV & Cover Letter for This Job
                      </button>
                    )}

                    {/* Loading state */}
                    {isGeneratingPreview && (
                      <div className="preview-loading">
                        <Loader2 className="spin" size={48} style={{ color: 'var(--primary)', marginBottom: '1rem' }} />
                        <h4>Analyzing Job & Tailoring Your CV...</h4>
                        <p className="text-muted">Reading job description and generating optimized CV and cover letter.</p>
                      </div>
                    )}

                    {/* Tailored documents preview */}
                    {tailoredPreview && !isGeneratingPreview && (
                      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                        <div className="preview-grid mb-4">
                          <div className="preview-pane">
                            <div className="pane-header"><FileText size={16} /> Tailored CV for This Job</div>
                            {renderTailoredResume(tailoredPreview.cv)}
                          </div>
                          <div className="preview-pane">
                            <div className="pane-header"><FileText size={16} /> Cover Letter</div>
                            <div className="cover-letter-formatted">
                              {tailoredPreview.cover_letter && tailoredPreview.cover_letter.trim()
                                ? tailoredPreview.cover_letter.split('\n').map((line: string, i: number) => (
                                    <p key={i} style={{ minHeight: line.trim() ? undefined : '0.75rem' }}>{line}</p>
                                  ))
                                : <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                    Cover letter could not be generated. Click <strong>Re-generate</strong> to try again.
                                  </p>
                              }
                            </div>
                          </div>
                        </div>

                        {/* Re-generate + Apply buttons */}
                        <div className="flex gap-2 mb-4">
                          <button className="btn btn-outline" onClick={handleGenerateTailoredDocs} disabled={isGeneratingPreview || appStatus.running}>
                            <Sparkles size={16} /> Re-generate
                          </button>
                        </div>

                        {!appStatus.running && !waitingForReviewCompletion && (
                          <button className="btn btn-success apply-btn" onClick={handleOpenApplyForm} disabled={appStatus.running}>
                            <Send size={20} /> Apply for This Job
                          </button>
                        )}
                      </motion.div>
                    )}

                    {/* Review banner - form is open */}
                    {appStatus.running && (
                      <div className="review-banner active">
                        <div className="flex items-center gap-4">
                          <div className="progress-circle">
                            <span className="text-xs font-bold">{appStatus.current_idx}/{appStatus.total}</span>
                          </div>
                          <div>
                            <strong>Form auto-filled for {appStatus.current_job?.title}</strong>
                            <p>Review the browser form, make adjustments, and submit. Then click below.</p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button className="btn btn-outline text-danger border-danger" onClick={handleSkipApplication}><SkipForward size={18} /> Skip</button>
                          <button className="btn btn-success" onClick={handleReviewComplete}><CheckCircle size={18} /> Submit Done</button>
                        </div>
                      </div>
                    )}

                    {waitingForReviewCompletion && (
                      <div className="review-banner">
                        <div className="flex items-center gap-4">
                          <Loader2 className="spin" size={24} style={{ color: 'var(--primary)' }} />
                          <div>
                            <strong>Finishing application...</strong>
                            <p>Saving to applied jobs and returning to overview.</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
