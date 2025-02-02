import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Divider,
  Stack,
  Button,
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { trainingApi } from '@/api/client';
import JobStatusChip from '@/components/JobStatusChip';
import JobProgress from '@/components/JobProgress';
import MetricsChart from '@/components/MetricsChart';

export default function JobDetails() {
  const { jobId } = useParams<{ jobId: string }>();
  const queryClient = useQueryClient();
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch job details
  const { data: job } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => trainingApi.getJob(jobId!),
    refetchInterval: autoRefresh ? 5000 : false,
    enabled: !!jobId,
  });

  // Stop job mutation
  const { mutate: stopJob } = useMutation({
    mutationFn: trainingApi.stopTraining,
    onSuccess: () => {
      queryClient.invalidateQueries(['job', jobId]);
    },
  });

  const handleStopJob = () => {
    if (confirm('Are you sure you want to stop this job?')) {
      stopJob(jobId!);
    }
  };

  if (!job) {
    return null;
  }

  const isRunning = job.status === 'running';
  const canStop = isRunning;

  // Prepare metrics data for charts
  const lossData = {
    labels: job.metrics.map((m) => format(new Date(m.timestamp), 'HH:mm:ss')),
    datasets: [
      {
        label: 'Training Loss',
        data: job.metrics.map((m) => m.train_loss),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
      {
        label: 'Validation Loss',
        data: job.metrics.map((m) => m.val_loss),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
      },
    ],
  };

  const accuracyData = {
    labels: job.metrics.map((m) => format(new Date(m.timestamp), 'HH:mm:ss')),
    datasets: [
      {
        label: 'Training Accuracy',
        data: job.metrics.map((m) => m.train_accuracy),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
      {
        label: 'Validation Accuracy',
        data: job.metrics.map((m) => m.val_accuracy),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
      },
    ],
  };

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h5">Training Job Details</Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? 'Pause Updates' : 'Resume Updates'}
          </Button>
          {canStop && (
            <Button
              variant="contained"
              color="error"
              onClick={handleStopJob}
            >
              Stop Job
            </Button>
          )}
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {/* Basic Info */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Job ID
                  </Typography>
                  <Typography variant="body1">{job.id}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Model ID
                  </Typography>
                  <Typography variant="body1">{job.model_id}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Status
                  </Typography>
                  <JobStatusChip status={job.status} />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Progress
                  </Typography>
                  <JobProgress progress={job.progress} />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Start Time
                  </Typography>
                  <Typography variant="body1">
                    {format(new Date(job.start_time), 'yyyy-MM-dd HH:mm:ss')}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="subtitle2" color="text.secondary">
                    End Time
                  </Typography>
                  <Typography variant="body1">
                    {job.end_time
                      ? format(new Date(job.end_time), 'yyyy-MM-dd HH:mm:ss')
                      : '-'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Config */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Configuration
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    CPU Cores
                  </Typography>
                  <Typography variant="body1">{job.config.cpu_cores}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Memory (MB)
                  </Typography>
                  <Typography variant="body1">{job.config.memory_mb}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    GPU Devices
                  </Typography>
                  <Typography variant="body1">{job.config.gpu_devices}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Batch Size
                  </Typography>
                  <Typography variant="body1">{job.config.batch_size}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Learning Rate
                  </Typography>
                  <Typography variant="body1">
                    {job.config.learning_rate}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Max Epochs
                  </Typography>
                  <Typography variant="body1">{job.config.max_epochs}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Logs */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Logs
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Box
                sx={{
                  height: 200,
                  overflow: 'auto',
                  bgcolor: 'background.paper',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                }}
              >
                {job.logs.map((log, index) => (
                  <Box key={index} sx={{ mb: 0.5 }}>
                    <Typography
                      variant="body2"
                      component="span"
                      color="text.secondary"
                      sx={{ mr: 1 }}
                    >
                      {format(new Date(log.timestamp), 'HH:mm:ss')}
                    </Typography>
                    <Typography variant="body2" component="span">
                      {log.message}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Loss Curves
              </Typography>
              <MetricsChart
                data={lossData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'Loss',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Accuracy Curves
              </Typography>
              <MetricsChart
                data={accuracyData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                      title: {
                        display: true,
                        text: 'Accuracy',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
} 