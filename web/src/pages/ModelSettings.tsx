import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Paper,
  Stack,
  Button,
  TextField,
  FormControl,
  FormControlLabel,
  FormGroup,
  Switch,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Alert,
  Snackbar,
  Divider,
} from '@mui/material';
import {
  Save as SaveIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/api/client';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface DeleteModelDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  modelName: string;
}

function DeleteModelDialog({
  open,
  onClose,
  onConfirm,
  modelName,
}: DeleteModelDialogProps) {
  const [confirmName, setConfirmName] = useState('');

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Delete Model</DialogTitle>
      <DialogContent>
        <DialogContentText>
          This action cannot be undone. This will permanently delete the model{' '}
          <strong>{modelName}</strong>, its versions, and all associated data.
        </DialogContentText>
        <DialogContentText sx={{ mt: 2 }}>
          Please type <strong>{modelName}</strong> to confirm.
        </DialogContentText>
        <TextField
          autoFocus
          fullWidth
          value={confirmName}
          onChange={(e) => setConfirmName(e.target.value)}
          margin="dense"
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={onConfirm}
          disabled={confirmName !== modelName}
          color="error"
          variant="contained"
        >
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelSettings() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tabValue, setTabValue] = useState(0);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Update model settings mutation
  const { mutate: updateSettings } = useMutation({
    mutationFn: (data: any) => modelApi.updateSettings(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId]);
      setMessage('Settings updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update settings');
      setSeverity('error');
    },
  });

  // Delete model mutation
  const { mutate: deleteModel } = useMutation({
    mutationFn: () => modelApi.deleteModel(modelId!),
    onSuccess: () => {
      navigate('/models');
    },
    onError: () => {
      setMessage('Failed to delete model');
      setSeverity('error');
    },
  });

  if (!model) {
    return null;
  }

  const handleSaveBasicSettings = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    updateSettings({
      name: formData.get('name'),
      description: formData.get('description'),
      visibility: formData.get('visibility'),
      tags: formData.get('tags')?.toString().split(',').map((tag) => tag.trim()),
    });
  };

  const handleSaveAdvancedSettings = (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    updateSettings({
      cache_enabled: formData.get('cache_enabled') === 'true',
      cache_ttl: parseInt(formData.get('cache_ttl') as string),
      max_batch_size: parseInt(formData.get('max_batch_size') as string),
      timeout: parseInt(formData.get('timeout') as string),
    });
  };

  const handleSaveNotificationSettings = (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    updateSettings({
      notifications: {
        email_enabled: formData.get('email_enabled') === 'true',
        email_recipients: formData
          .get('email_recipients')
          ?.toString()
          .split(',')
          .map((email) => email.trim()),
        slack_enabled: formData.get('slack_enabled') === 'true',
        slack_channel: formData.get('slack_channel'),
      },
    });
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Settings
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Configure settings for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            aria-label="settings tabs"
          >
            <Tab
              icon={<SettingsIcon />}
              label="Basic"
              id="settings-tab-0"
              aria-controls="settings-tabpanel-0"
            />
            <Tab
              icon={<StorageIcon />}
              label="Advanced"
              id="settings-tab-1"
              aria-controls="settings-tabpanel-1"
            />
            <Tab
              icon={<NotificationsIcon />}
              label="Notifications"
              id="settings-tab-2"
              aria-controls="settings-tabpanel-2"
            />
          </Tabs>

          {/* Basic Settings */}
          <TabPanel value={tabValue} index={0}>
            <Paper variant="outlined" sx={{ p: 3 }}>
              <form onSubmit={handleSaveBasicSettings}>
                <Stack spacing={3}>
                  <TextField
                    name="name"
                    label="Model Name"
                    defaultValue={model.name}
                    fullWidth
                    required
                  />
                  <TextField
                    name="description"
                    label="Description"
                    defaultValue={model.description}
                    multiline
                    rows={3}
                    fullWidth
                  />
                  <FormControl fullWidth>
                    <InputLabel>Visibility</InputLabel>
                    <Select
                      name="visibility"
                      defaultValue={model.visibility}
                      label="Visibility"
                    >
                      <MenuItem value="private">Private</MenuItem>
                      <MenuItem value="team">Team</MenuItem>
                      <MenuItem value="public">Public</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    name="tags"
                    label="Tags"
                    defaultValue={model.tags?.join(', ')}
                    helperText="Separate tags with commas"
                    fullWidth
                  />
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<SaveIcon />}
                    >
                      Save Changes
                    </Button>
                  </Box>
                </Stack>
              </form>
            </Paper>
          </TabPanel>

          {/* Advanced Settings */}
          <TabPanel value={tabValue} index={1}>
            <Paper variant="outlined" sx={{ p: 3 }}>
              <form onSubmit={handleSaveAdvancedSettings}>
                <Stack spacing={3}>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch
                          name="cache_enabled"
                          defaultChecked={model.cache_enabled}
                        />
                      }
                      label="Enable Response Caching"
                    />
                  </FormGroup>
                  <TextField
                    name="cache_ttl"
                    label="Cache TTL (seconds)"
                    type="number"
                    defaultValue={model.cache_ttl}
                    disabled={!model.cache_enabled}
                    fullWidth
                  />
                  <TextField
                    name="max_batch_size"
                    label="Maximum Batch Size"
                    type="number"
                    defaultValue={model.max_batch_size}
                    helperText="Maximum number of inputs to process in a single batch"
                    fullWidth
                  />
                  <TextField
                    name="timeout"
                    label="Request Timeout (seconds)"
                    type="number"
                    defaultValue={model.timeout}
                    fullWidth
                  />
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<SaveIcon />}
                    >
                      Save Changes
                    </Button>
                  </Box>
                </Stack>
              </form>
            </Paper>
          </TabPanel>

          {/* Notification Settings */}
          <TabPanel value={tabValue} index={2}>
            <Paper variant="outlined" sx={{ p: 3 }}>
              <form onSubmit={handleSaveNotificationSettings}>
                <Stack spacing={3}>
                  <Typography variant="subtitle2" gutterBottom>
                    Email Notifications
                  </Typography>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch
                          name="email_enabled"
                          defaultChecked={model.notifications?.email_enabled}
                        />
                      }
                      label="Enable Email Notifications"
                    />
                  </FormGroup>
                  <TextField
                    name="email_recipients"
                    label="Email Recipients"
                    defaultValue={model.notifications?.email_recipients?.join(', ')}
                    helperText="Separate email addresses with commas"
                    disabled={!model.notifications?.email_enabled}
                    fullWidth
                  />

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" gutterBottom>
                    Slack Notifications
                  </Typography>
                  <FormGroup>
                    <FormControlLabel
                      control={
                        <Switch
                          name="slack_enabled"
                          defaultChecked={model.notifications?.slack_enabled}
                        />
                      }
                      label="Enable Slack Notifications"
                    />
                  </FormGroup>
                  <TextField
                    name="slack_channel"
                    label="Slack Channel"
                    defaultValue={model.notifications?.slack_channel}
                    helperText="Channel name without #"
                    disabled={!model.notifications?.slack_enabled}
                    fullWidth
                  />

                  <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<SaveIcon />}
                    >
                      Save Changes
                    </Button>
                  </Box>
                </Stack>
              </form>
            </Paper>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Danger Zone */}
      <Card sx={{ mt: 3, bgcolor: 'error.main' }}>
        <CardContent>
          <Typography variant="h6" color="white" gutterBottom>
            Danger Zone
          </Typography>
          <Paper sx={{ p: 3 }}>
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
            >
              <div>
                <Typography variant="subtitle1">Delete Model</Typography>
                <Typography variant="body2" color="text.secondary">
                  Once you delete a model, there is no going back. Please be
                  certain.
                </Typography>
              </div>
              <Button
                variant="contained"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => setDeleteDialogOpen(true)}
              >
                Delete Model
              </Button>
            </Stack>
          </Paper>
        </CardContent>
      </Card>

      {/* Delete Dialog */}
      <DeleteModelDialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        onConfirm={() => {
          setDeleteDialogOpen(false);
          deleteModel();
        }}
        modelName={model.name}
      />

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