import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Stack,
  Alert,
  Snackbar,
  InputAdornment,
} from '@mui/material';
import { LoadingButton } from '@mui/lab';
import { useQuery, useMutation } from '@tanstack/react-query';
import { systemApi } from '@/api/client';

interface SettingsSectionProps {
  title: string;
  description?: string;
  children: React.ReactNode;
}

function SettingsSection({ title, description, children }: SettingsSectionProps) {
  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        {description && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 3 }}
          >
            {description}
          </Typography>
        )}
        {children}
      </CardContent>
    </Card>
  );
}

export default function Settings() {
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch settings
  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: systemApi.getSettings,
  });

  // Update settings mutation
  const { mutate: updateSettings, isLoading: isUpdating } = useMutation({
    mutationFn: systemApi.updateSettings,
    onSuccess: () => {
      setMessage('Settings updated successfully');
      setSeverity('success');
    },
    onError: (error) => {
      setMessage('Failed to update settings');
      setSeverity('error');
    },
  });

  // Test connection mutation
  const { mutate: testConnection, isLoading: isTesting } = useMutation({
    mutationFn: systemApi.testConnection,
    onSuccess: () => {
      setMessage('Connection test successful');
      setSeverity('success');
    },
    onError: (error) => {
      setMessage('Connection test failed');
      setSeverity('error');
    },
  });

  if (isLoading || !settings) {
    return null;
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateSettings(settings);
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h5">Settings</Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            onClick={() => testConnection()}
            disabled={isTesting}
          >
            {isTesting ? 'Testing...' : 'Test Connection'}
          </Button>
          <LoadingButton
            type="submit"
            variant="contained"
            loading={isUpdating}
          >
            Save Changes
          </LoadingButton>
        </Stack>
      </Box>

      {/* System Settings */}
      <SettingsSection
        title="System Settings"
        description="Configure basic system parameters and paths"
      >
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              label="Data Directory"
              value={settings.data_dir}
              onChange={(e) =>
                updateSettings({ ...settings, data_dir: e.target.value })
              }
              fullWidth
              required
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Temp Directory"
              value={settings.temp_dir}
              onChange={(e) =>
                updateSettings({ ...settings, temp_dir: e.target.value })
              }
              fullWidth
              required
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Log Level</InputLabel>
              <Select
                value={settings.log_level}
                label="Log Level"
                onChange={(e) =>
                  updateSettings({ ...settings, log_level: e.target.value })
                }
              >
                <MenuItem value="debug">Debug</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Max Log Size"
              value={settings.max_log_size}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  max_log_size: Number(e.target.value),
                })
              }
              type="number"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">MB</InputAdornment>
                ),
              }}
              fullWidth
            />
          </Grid>
        </Grid>
      </SettingsSection>

      {/* Training Settings */}
      <SettingsSection
        title="Training Settings"
        description="Configure default training parameters"
      >
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              label="Default Batch Size"
              value={settings.default_batch_size}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  default_batch_size: Number(e.target.value),
                })
              }
              type="number"
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Default Learning Rate"
              value={settings.default_learning_rate}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  default_learning_rate: Number(e.target.value),
                })
              }
              type="number"
              inputProps={{ step: 0.0001 }}
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Default Max Epochs"
              value={settings.default_max_epochs}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  default_max_epochs: Number(e.target.value),
                })
              }
              type="number"
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.enable_early_stopping}
                  onChange={(e) =>
                    updateSettings({
                      ...settings,
                      enable_early_stopping: e.target.checked,
                    })
                  }
                />
              }
              label="Enable Early Stopping"
            />
          </Grid>
        </Grid>
      </SettingsSection>

      {/* Resource Settings */}
      <SettingsSection
        title="Resource Settings"
        description="Configure system resource limits"
      >
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              label="Max CPU Cores Per Job"
              value={settings.max_cpu_cores_per_job}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  max_cpu_cores_per_job: Number(e.target.value),
                })
              }
              type="number"
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Max Memory Per Job"
              value={settings.max_memory_per_job}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  max_memory_per_job: Number(e.target.value),
                })
              }
              type="number"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">MB</InputAdornment>
                ),
              }}
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Max GPU Devices Per Job"
              value={settings.max_gpu_devices_per_job}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  max_gpu_devices_per_job: Number(e.target.value),
                })
              }
              type="number"
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Max Concurrent Jobs"
              value={settings.max_concurrent_jobs}
              onChange={(e) =>
                updateSettings({
                  ...settings,
                  max_concurrent_jobs: Number(e.target.value),
                })
              }
              type="number"
              fullWidth
            />
          </Grid>
        </Grid>
      </SettingsSection>

      {/* UI Settings */}
      <SettingsSection
        title="UI Settings"
        description="Configure user interface preferences"
      >
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Theme</InputLabel>
              <Select
                value={settings.theme}
                label="Theme"
                onChange={(e) =>
                  updateSettings({ ...settings, theme: e.target.value })
                }
              >
                <MenuItem value="light">Light</MenuItem>
                <MenuItem value="dark">Dark</MenuItem>
                <MenuItem value="system">System</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Date Format</InputLabel>
              <Select
                value={settings.date_format}
                label="Date Format"
                onChange={(e) =>
                  updateSettings({ ...settings, date_format: e.target.value })
                }
              >
                <MenuItem value="yyyy-MM-dd HH:mm:ss">
                  YYYY-MM-DD HH:mm:ss
                </MenuItem>
                <MenuItem value="MM/dd/yyyy HH:mm:ss">
                  MM/DD/YYYY HH:mm:ss
                </MenuItem>
                <MenuItem value="dd/MM/yyyy HH:mm:ss">
                  DD/MM/YYYY HH:mm:ss
                </MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.enable_animations}
                  onChange={(e) =>
                    updateSettings({
                      ...settings,
                      enable_animations: e.target.checked,
                    })
                  }
                />
              }
              label="Enable Animations"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.enable_notifications}
                  onChange={(e) =>
                    updateSettings({
                      ...settings,
                      enable_notifications: e.target.checked,
                    })
                  }
                />
              }
              label="Enable Notifications"
            />
          </Grid>
        </Grid>
      </SettingsSection>

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