import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TrainingConfig {
  model_id: string;
  cpu_cores?: number;
  memory_mb?: number;
  gpu_devices?: number;
  batch_size?: number;
  learning_rate?: number;
  max_epochs?: number;
}

export interface JobResponse {
  id: string;
  model_id: string;
  status: string;
  progress: number;
  start_time: string;
  end_time?: string;
  error?: string;
}

export interface JobList {
  jobs: JobResponse[];
  total: number;
}

export interface SystemMetrics {
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  gpu_utilization?: number[];
  gpu_memory_used?: number[];
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
  };
}

export interface TrainingMetrics {
  timestamp: string;
  job_id: string;
  epoch: number;
  loss: number;
  accuracy: number;
  learning_rate: number;
  batch_size: number;
  samples_processed: number;
  time_elapsed: number;
  custom_metrics?: Record<string, number>;
}

export interface TrainingSummary {
  total_epochs: number;
  avg_loss: number;
  best_accuracy: number;
  total_time: number;
  samples_per_second: number;
}

export const trainingApi = {
  // Training jobs
  startTraining: async (config: TrainingConfig) => {
    const response = await apiClient.post<JobResponse>('/training/jobs', config);
    return response.data;
  },

  stopTraining: async (jobId: string) => {
    const response = await apiClient.delete(`/training/jobs/${jobId}`);
    return response.data;
  },

  getJobProgress: async (jobId: string) => {
    const response = await apiClient.get<JobResponse>(`/training/jobs/${jobId}`);
    return response.data;
  },

  listJobs: async (status?: string) => {
    const response = await apiClient.get<JobList>('/training/jobs', {
      params: { status },
    });
    return response.data;
  },

  getJobLogs: async (jobId: string) => {
    const response = await apiClient.get<string[]>(`/training/jobs/${jobId}/logs`);
    return response.data;
  },

  cleanupJobs: async (maxAgeDays: number = 7) => {
    const response = await apiClient.post('/training/cleanup', null, {
      params: { max_age_days: maxAgeDays },
    });
    return response.data;
  },

  // System metrics
  getSystemMetrics: async (startTime?: string, endTime?: string) => {
    const response = await apiClient.get<SystemMetrics[]>('/metrics/system', {
      params: { start_time: startTime, end_time: endTime },
    });
    return response.data;
  },

  // Training metrics
  getTrainingMetrics: async (
    jobId: string,
    startTime?: string,
    endTime?: string
  ) => {
    const response = await apiClient.get<TrainingMetrics[]>(
      `/metrics/training/${jobId}`,
      {
        params: { start_time: startTime, end_time: endTime },
      }
    );
    return response.data;
  },

  getTrainingSummary: async (jobId: string) => {
    const response = await apiClient.get<TrainingSummary>(
      `/metrics/training/${jobId}/summary`
    );
    return response.data;
  },
};

export default apiClient; 