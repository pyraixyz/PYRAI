import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Stack,
  Button,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Snackbar,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  DateRange as DateRangeIcon,
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Group as GroupIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format, subDays } from 'date-fns';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { modelApi } from '@/api/client';

interface AnalyticsData {
  summary: {
    total_requests: number;
    average_latency: number;
    error_rate: number;
    success_rate: number;
    total_compute_time: number;
    total_tokens: number;
    unique_users: number;
  };
  requests_over_time: {
    timestamp: string;
    requests: number;
    errors: number;
    latency: number;
  }[];
  top_users: {
    user_id: string;
    user_name: string;
    request_count: number;
    error_count: number;
  }[];
  error_types: {
    error_type: string;
    count: number;
    percentage: number;
  }[];
  resource_usage: {
    timestamp: string;
    cpu_usage: number;
    memory_usage: number;
    gpu_usage?: number;
  }[];
}

const TIME_RANGES = [
  { label: 'Last 24 Hours', value: 1 },
  { label: 'Last 7 Days', value: 7 },
  { label: 'Last 30 Days', value: 30 },
  { label: 'Last 90 Days', value: 90 },
];

const COLORS = [
  '#2196f3',
  '#4caf50',
  '#f44336',
  '#ff9800',
  '#9c27b0',
  '#00bcd4',
  '#ffeb3b',
  '#795548',
];

function MetricCard({
  title,
  value,
  change,
  icon,
  color,
}: {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color?: string;
}) {
  return (
    <Card>
      <CardContent>
        <Stack spacing={1}>
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            justifyContent="space-between"
          >
            <Typography variant="subtitle2" color="text.secondary">
              {title}
            </Typography>
            {icon}
          </Stack>
          <Typography variant="h4">{value}</Typography>
          {change !== undefined && (
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip
                size="small"
                label={`${change >= 0 ? '+' : ''}${change}%`}
                color={change >= 0 ? 'success' : 'error'}
              />
              <Typography variant="caption" color="text.secondary">
                vs previous period
              </Typography>
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function ModelAnalytics() {
  const { modelId } = useParams<{ modelId: string }>();
  const [timeRange, setTimeRange] = useState(7);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch analytics data
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['model', modelId, 'analytics', timeRange],
    queryFn: () =>
      modelApi.getModelAnalytics(modelId!, {
        start_date: format(subDays(new Date(), timeRange), 'yyyy-MM-dd'),
        end_date: format(new Date(), 'yyyy-MM-dd'),
      }),
    enabled: !!modelId,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (!model || (!analytics && !isLoading)) {
    return null;
  }

  const handleExport = async () => {
    try {
      await modelApi.exportModelAnalytics(modelId!, {
        start_date: format(subDays(new Date(), timeRange), 'yyyy-MM-dd'),
        end_date: format(new Date(), 'yyyy-MM-dd'),
      });
      setMessage('Analytics data exported successfully');
      setSeverity('success');
    } catch (error) {
      setMessage('Failed to export analytics data');
      setSeverity('error');
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Analytics
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Performance analytics and usage statistics for {model.name}
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
        <FormControl size="small" sx={{ width: 200 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value as number)}
            startAdornment={<DateRangeIcon sx={{ mr: 1 }} />}
          >
            {TIME_RANGES.map((range) => (
              <MenuItem key={range.value} value={range.value}>
                {range.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
          >
            Export
          </Button>
          <Tooltip title="Refresh">
            <IconButton>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {isLoading ? (
        <LinearProgress />
      ) : (
        analytics && (
          <Stack spacing={3}>
            {/* Summary Metrics */}
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Total Requests"
                  value={analytics.summary.total_requests.toLocaleString()}
                  icon={<AssessmentIcon color="primary" />}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Average Latency"
                  value={`${analytics.summary.average_latency.toFixed(2)}ms`}
                  icon={<SpeedIcon color="warning" />}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Success Rate"
                  value={`${(analytics.summary.success_rate * 100).toFixed(1)}%`}
                  icon={<TrendingUpIcon color="success" />}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                  title="Unique Users"
                  value={analytics.summary.unique_users.toLocaleString()}
                  icon={<GroupIcon color="info" />}
                />
              </Grid>
            </Grid>

            {/* Request Trends */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Request Trends
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={analytics.requests_over_time}
                      margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(value) =>
                          format(new Date(value), 'MMM d, HH:mm')
                        }
                      />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <RechartsTooltip
                        formatter={(value: number, name: string) => [
                          value.toLocaleString(),
                          name,
                        ]}
                        labelFormatter={(label: string) =>
                          format(new Date(label), 'MMM d, HH:mm')
                        }
                      />
                      <Legend />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="requests"
                        name="Requests"
                        stroke={COLORS[0]}
                        activeDot={{ r: 8 }}
                      />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="errors"
                        name="Errors"
                        stroke={COLORS[2]}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="latency"
                        name="Latency (ms)"
                        stroke={COLORS[1]}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>

            {/* Resource Usage */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Resource Usage
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={analytics.resource_usage}
                      margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(value) =>
                          format(new Date(value), 'MMM d, HH:mm')
                        }
                      />
                      <YAxis />
                      <RechartsTooltip
                        formatter={(value: number, name: string) => [
                          `${value.toFixed(1)}%`,
                          name,
                        ]}
                        labelFormatter={(label: string) =>
                          format(new Date(label), 'MMM d, HH:mm')
                        }
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="cpu_usage"
                        name="CPU Usage"
                        stroke={COLORS[0]}
                      />
                      <Line
                        type="monotone"
                        dataKey="memory_usage"
                        name="Memory Usage"
                        stroke={COLORS[1]}
                      />
                      {analytics.resource_usage[0]?.gpu_usage !== undefined && (
                        <Line
                          type="monotone"
                          dataKey="gpu_usage"
                          name="GPU Usage"
                          stroke={COLORS[2]}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>

            {/* Error Distribution */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Error Distribution
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={analytics.error_types}
                            dataKey="count"
                            nameKey="error_type"
                            cx="50%"
                            cy="50%"
                            outerRadius={100}
                            label={(entry) =>
                              `${entry.error_type} (${entry.percentage}%)`
                            }
                          >
                            {analytics.error_types.map((entry, index) => (
                              <Cell
                                key={entry.error_type}
                                fill={COLORS[index % COLORS.length]}
                              />
                            ))}
                          </Pie>
                          <RechartsTooltip
                            formatter={(value: number, name: string) => [
                              value.toLocaleString(),
                              name,
                            ]}
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Top Users
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={analytics.top_users}
                          margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="user_name" />
                          <YAxis />
                          <RechartsTooltip
                            formatter={(value: number, name: string) => [
                              value.toLocaleString(),
                              name,
                            ]}
                          />
                          <Legend />
                          <Bar
                            dataKey="request_count"
                            name="Requests"
                            fill={COLORS[0]}
                          />
                          <Bar
                            dataKey="error_count"
                            name="Errors"
                            fill={COLORS[2]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Stack>
        )
      )}

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