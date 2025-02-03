import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Button,
  Paper,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  IconButton,
  Chip,
  Alert,
  Snackbar,
  LinearProgress,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as HealthyIcon,
  Error as ErrorIcon,
  Timeline as TimelineIcon,
  Speed as PerformanceIcon,
  Memory as ResourceIcon,
  Assessment as AccuracyIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { modelApi } from '@/api/client';

interface MetricCard {
  title: string;
  value: string | number;
  change?: number;
  status?: 'success' | 'warning' | 'error';
  icon: React.ReactNode;
}

interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

function MetricCard({ title, value, change, status, icon }: MetricCard) {
  return (
    <Card>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center">
          <Box sx={{ color: 'primary.main' }}>{icon}</Box>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" gutterBottom>
              {value}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {title}
            </Typography>
            {change !== undefined && (
              <Typography
                variant="caption"
                color={change >= 0 ? 'success.main' : 'error.main'}
              >
                {change >= 0 ? '+' : ''}
                {change}% vs last period
              </Typography>
            )}
          </Box>
          {status && (
            <Box>
              {status === 'success' && (
                <HealthyIcon color="success" fontSize="small" />
              )}
              {status === 'warning' && (
                <WarningIcon color="warning" fontSize="small" />
              )}
              {status === 'error' && <ErrorIcon color="error" fontSize="small" />}
            </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function ModelMonitoring() {
  const { modelId } = useParams<{ modelId: string }>();
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch monitoring data
  const { data: monitoring, refetch: refetchMonitoring } = useQuery({
    queryKey: ['model', modelId, 'monitoring', timeRange],
    queryFn: () => modelApi.getMonitoring(modelId!, { timeRange }),
    enabled: !!modelId,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch alerts
  const { data: alerts } = useQuery({
    queryKey: ['model', modelId, 'alerts'],
    queryFn: () => modelApi.getAlerts(modelId!),
    enabled: !!modelId,
    refetchInterval: 30000,
  });

  if (!model || !monitoring || !alerts) {
    return null;
  }

  const {
    health,
    performance,
    resources,
    accuracy,
    predictions,
    latencyData,
    throughputData,
    resourceUsageData,
    accuracyData,
  } = monitoring;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Monitoring
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Monitor performance and health metrics for {model.name}
        </Typography>
      </Box>

      {/* Controls */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: 3 }}
      >
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) =>
              setTimeRange(e.target.value as '1h' | '24h' | '7d' | '30d')
            }
          >
            <MenuItem value="1h">Last Hour</MenuItem>
            <MenuItem value="24h">Last 24 Hours</MenuItem>
            <MenuItem value="7d">Last 7 Days</MenuItem>
            <MenuItem value="30d">Last 30 Days</MenuItem>
          </Select>
        </FormControl>

        <Tooltip title="Refresh">
          <IconButton onClick={() => refetchMonitoring()}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Stack spacing={1} sx={{ mb: 3 }}>
          {alerts.map((alert) => (
            <Alert
              key={alert.id}
              severity={alert.type}
              onClose={() => {
                // Handle alert acknowledgment
              }}
            >
              <Typography variant="body2">{alert.message}</Typography>
              <Typography variant="caption" color="text.secondary">
                {format(new Date(alert.timestamp), 'yyyy-MM-dd HH:mm:ss')}
              </Typography>
            </Alert>
          ))}
        </Stack>
      )}

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Health Status"
            value={health.status}
            status={health.status === 'Healthy' ? 'success' : 'error'}
            icon={<HealthyIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Average Latency"
            value={`${performance.avgLatency.toFixed(2)}ms`}
            change={performance.latencyChange}
            status={
              performance.avgLatency < 100
                ? 'success'
                : performance.avgLatency < 200
                ? 'warning'
                : 'error'
            }
            icon={<PerformanceIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Resource Usage"
            value={`${resources.cpuUsage.toFixed(1)}%`}
            change={resources.cpuChange}
            status={
              resources.cpuUsage < 70
                ? 'success'
                : resources.cpuUsage < 85
                ? 'warning'
                : 'error'
            }
            icon={<ResourceIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Accuracy"
            value={`${(accuracy.value * 100).toFixed(2)}%`}
            change={accuracy.change}
            status={
              accuracy.value > 0.95
                ? 'success'
                : accuracy.value > 0.9
                ? 'warning'
                : 'error'
            }
            icon={<AccuracyIcon />}
          />
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Latency Chart */}
        <Grid item xs={12} lg={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Latency Trends
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="p50"
                    stroke="#8884d8"
                    name="P50 Latency"
                  />
                  <Line
                    type="monotone"
                    dataKey="p95"
                    stroke="#82ca9d"
                    name="P95 Latency"
                  />
                  <Line
                    type="monotone"
                    dataKey="p99"
                    stroke="#ffc658"
                    name="P99 Latency"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Throughput Chart */}
        <Grid item xs={12} lg={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Request Throughput
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={throughputData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="requests"
                    stroke="#8884d8"
                    fill="#8884d8"
                    name="Requests/sec"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Resource Usage Chart */}
        <Grid item xs={12} lg={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Resource Usage
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={resourceUsageData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="cpu"
                    stroke="#8884d8"
                    name="CPU Usage (%)"
                  />
                  <Line
                    type="monotone"
                    dataKey="memory"
                    stroke="#82ca9d"
                    name="Memory Usage (%)"
                  />
                  <Line
                    type="monotone"
                    dataKey="gpu"
                    stroke="#ffc658"
                    name="GPU Usage (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Accuracy Chart */}
        <Grid item xs={12} lg={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Accuracy
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis domain={[0.8, 1]} />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#8884d8"
                    name="Accuracy"
                  />
                  <Line
                    type="monotone"
                    dataKey="f1_score"
                    stroke="#82ca9d"
                    name="F1 Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Message Snackbar */}
      <Snackbar
        open={!!message}
        autoHideDuration={3000}
        onClose={() => setMessage('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setMessage('')}
          severity={severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {message}
        </Alert>
      </Snackbar>
    </Box>
  );
} 