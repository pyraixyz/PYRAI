import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Stack,
  Button,
  TextField,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Divider,
  Alert,
  Snackbar,
  Tooltip,
  Chip,
  Rating,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Feedback as FeedbackIcon,
  Reply as ReplyIcon,
  Flag as FlagIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Star as StarIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface Feedback {
  id: string;
  type: 'bug' | 'feature' | 'improvement' | 'other';
  title: string;
  description: string;
  rating: number;
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high';
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
    email: string;
  };
  responses: Array<{
    id: string;
    content: string;
    created_at: string;
    created_by: {
      id: string;
      name: string;
      email: string;
    };
  }>;
}

interface FeedbackDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  feedback?: Feedback;
}

function FeedbackDialog({
  open,
  onClose,
  onSave,
  feedback,
}: FeedbackDialogProps) {
  const [formData, setFormData] = useState(
    feedback || {
      title: '',
      description: '',
      type: 'bug',
      rating: 0,
      priority: 'medium',
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {feedback ? 'Edit Feedback' : 'Submit Feedback'}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Title"
            value={formData.title}
            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
            fullWidth
            required
          />

          <FormControl fullWidth>
            <InputLabel>Type</InputLabel>
            <Select
              value={formData.type}
              label="Type"
              onChange={(e) =>
                setFormData({ ...formData, type: e.target.value })
              }
            >
              <MenuItem value="bug">Bug Report</MenuItem>
              <MenuItem value="feature">Feature Request</MenuItem>
              <MenuItem value="improvement">Improvement</MenuItem>
              <MenuItem value="other">Other</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>Priority</InputLabel>
            <Select
              value={formData.priority}
              label="Priority"
              onChange={(e) =>
                setFormData({ ...formData, priority: e.target.value })
              }
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
            </Select>
          </FormControl>

          <Box>
            <Typography component="legend">Rating</Typography>
            <Rating
              value={formData.rating}
              onChange={(_, newValue) =>
                setFormData({ ...formData, rating: newValue || 0 })
              }
            />
          </Box>

          <TextField
            label="Description"
            value={formData.description}
            onChange={(e) =>
              setFormData({ ...formData, description: e.target.value })
            }
            multiline
            rows={6}
            fullWidth
            required
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.title || !formData.description}
        >
          Submit
        </Button>
      </DialogActions>
    </Dialog>
  );
}

interface ResponseDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  feedback: Feedback;
}

function ResponseDialog({
  open,
  onClose,
  onSave,
  feedback,
}: ResponseDialogProps) {
  const [content, setContent] = useState('');

  const handleSave = () => {
    onSave({ content });
    setContent('');
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Add Response</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <TextField
            label="Response"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            multiline
            rows={4}
            fullWidth
            required
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained" disabled={!content}>
          Submit
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelFeedback() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [feedbackDialogOpen, setFeedbackDialogOpen] = useState(false);
  const [responseDialogOpen, setResponseDialogOpen] = useState(false);
  const [selectedFeedback, setSelectedFeedback] = useState<Feedback | undefined>();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch feedback
  const { data: feedback } = useQuery({
    queryKey: ['model', modelId, 'feedback'],
    queryFn: () => modelApi.getFeedback(modelId!),
    enabled: !!modelId,
  });

  // Create feedback mutation
  const { mutate: createFeedback } = useMutation({
    mutationFn: (data: any) => modelApi.createFeedback(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'feedback']);
      setMessage('Feedback submitted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to submit feedback');
      setSeverity('error');
    },
  });

  // Update feedback mutation
  const { mutate: updateFeedback } = useMutation({
    mutationFn: (data: any) =>
      modelApi.updateFeedback(modelId!, data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'feedback']);
      setMessage('Feedback updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update feedback');
      setSeverity('error');
    },
  });

  // Delete feedback mutation
  const { mutate: deleteFeedback } = useMutation({
    mutationFn: (feedbackId: string) =>
      modelApi.deleteFeedback(modelId!, feedbackId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'feedback']);
      setMessage('Feedback deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete feedback');
      setSeverity('error');
    },
  });

  // Add response mutation
  const { mutate: addResponse } = useMutation({
    mutationFn: (data: any) =>
      modelApi.addResponse(modelId!, selectedFeedback!.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'feedback']);
      setMessage('Response added successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to add response');
      setSeverity('error');
    },
  });

  if (!model || !feedback) {
    return null;
  }

  const handleSaveFeedback = (data: any) => {
    if (selectedFeedback) {
      updateFeedback({ ...data, id: selectedFeedback.id });
    } else {
      createFeedback(data);
    }
  };

  const handleDeleteFeedback = (feedbackId: string) => {
    if (confirm('Are you sure you want to delete this feedback?')) {
      deleteFeedback(feedbackId);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'bug':
        return 'error';
      case 'feature':
        return 'primary';
      case 'improvement':
        return 'success';
      default:
        return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open':
        return 'info';
      case 'in_progress':
        return 'warning';
      case 'resolved':
        return 'success';
      case 'closed':
        return 'default';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Feedback
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage feedback for {model.name}
        </Typography>
      </Box>

      {/* Actions */}
      <Box sx={{ mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedFeedback(undefined);
            setFeedbackDialogOpen(true);
          }}
        >
          Submit Feedback
        </Button>
      </Box>

      {/* Feedback List */}
      <Grid container spacing={3}>
        {feedback.map((item) => (
          <Grid item xs={12} key={item.id}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Stack spacing={2}>
                <Stack
                  direction="row"
                  spacing={2}
                  alignItems="center"
                  justifyContent="space-between"
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <FeedbackIcon />
                    <Typography variant="h6">{item.title}</Typography>
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedFeedback(item);
                          setFeedbackDialogOpen(true);
                        }}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteFeedback(item.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </Stack>

                <Stack direction="row" spacing={1}>
                  <Chip
                    label={item.type}
                    size="small"
                    color={getTypeColor(item.type)}
                  />
                  <Chip
                    label={item.priority}
                    size="small"
                    color={getPriorityColor(item.priority)}
                  />
                  <Chip
                    label={item.status}
                    size="small"
                    color={getStatusColor(item.status)}
                  />
                  <Rating value={item.rating} readOnly size="small" />
                </Stack>

                <Typography variant="body1">{item.description}</Typography>

                <Typography variant="caption" color="text.secondary">
                  Submitted by {item.created_by.name} on{' '}
                  {format(new Date(item.created_at), 'yyyy-MM-dd HH:mm')}
                </Typography>

                {item.responses.length > 0 && (
                  <Box sx={{ pl: 2, borderLeft: 1, borderColor: 'divider' }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Responses
                    </Typography>
                    <Stack spacing={2}>
                      {item.responses.map((response) => (
                        <Paper
                          key={response.id}
                          variant="outlined"
                          sx={{ p: 1 }}
                        >
                          <Typography variant="body2">
                            {response.content}
                          </Typography>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            display="block"
                          >
                            {response.created_by.name} -{' '}
                            {format(
                              new Date(response.created_at),
                              'yyyy-MM-dd HH:mm'
                            )}
                          </Typography>
                        </Paper>
                      ))}
                    </Stack>
                  </Box>
                )}

                <Box>
                  <Button
                    startIcon={<ReplyIcon />}
                    onClick={() => {
                      setSelectedFeedback(item);
                      setResponseDialogOpen(true);
                    }}
                  >
                    Add Response
                  </Button>
                </Box>
              </Stack>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Feedback Dialog */}
      <FeedbackDialog
        open={feedbackDialogOpen}
        onClose={() => {
          setFeedbackDialogOpen(false);
          setSelectedFeedback(undefined);
        }}
        onSave={handleSaveFeedback}
        feedback={selectedFeedback}
      />

      {/* Response Dialog */}
      {selectedFeedback && (
        <ResponseDialog
          open={responseDialogOpen}
          onClose={() => {
            setResponseDialogOpen(false);
            setSelectedFeedback(undefined);
          }}
          onSave={addResponse}
          feedback={selectedFeedback}
        />
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