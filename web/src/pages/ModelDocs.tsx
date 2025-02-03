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
  Tab,
  Tabs,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Description as DocIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Preview as PreviewIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { modelApi } from '@/api/client';

interface Document {
  id: string;
  title: string;
  content: string;
  type: 'markdown' | 'pdf' | 'image';
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

interface DocumentDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: any) => void;
  document?: Document;
}

function DocumentDialog({
  open,
  onClose,
  onSave,
  document,
}: DocumentDialogProps) {
  const [formData, setFormData] = useState(
    document || {
      title: '',
      content: '',
      type: 'markdown',
    }
  );
  const [previewMode, setPreviewMode] = useState(false);

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        {document ? 'Edit Document' : 'Create Document'}
        {formData.type === 'markdown' && (
          <Button
            sx={{ float: 'right' }}
            onClick={() => setPreviewMode(!previewMode)}
            startIcon={previewMode ? <EditIcon /> : <PreviewIcon />}
          >
            {previewMode ? 'Edit' : 'Preview'}
          </Button>
        )}
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

          {formData.type === 'markdown' && !previewMode && (
            <TextField
              label="Content"
              value={formData.content}
              onChange={(e) =>
                setFormData({ ...formData, content: e.target.value })
              }
              multiline
              rows={20}
              fullWidth
              required
            />
          )}

          {formData.type === 'markdown' && previewMode && (
            <Paper
              variant="outlined"
              sx={{ p: 2, minHeight: '500px', overflow: 'auto' }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {formData.content}
              </ReactMarkdown>
            </Paper>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.title || !formData.content}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}

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
      id={`docs-tabpanel-${index}`}
      aria-labelledby={`docs-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function ModelDocs() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [tabValue, setTabValue] = useState(0);
  const [documentDialogOpen, setDocumentDialogOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | undefined>();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch documents
  const { data: documents } = useQuery({
    queryKey: ['model', modelId, 'documents'],
    queryFn: () => modelApi.getDocuments(modelId!),
    enabled: !!modelId,
  });

  // Create document mutation
  const { mutate: createDocument } = useMutation({
    mutationFn: (data: any) => modelApi.createDocument(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'documents']);
      setMessage('Document created successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to create document');
      setSeverity('error');
    },
  });

  // Update document mutation
  const { mutate: updateDocument } = useMutation({
    mutationFn: (data: any) =>
      modelApi.updateDocument(modelId!, data.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'documents']);
      setMessage('Document updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update document');
      setSeverity('error');
    },
  });

  // Delete document mutation
  const { mutate: deleteDocument } = useMutation({
    mutationFn: (documentId: string) =>
      modelApi.deleteDocument(modelId!, documentId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'documents']);
      setMessage('Document deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete document');
      setSeverity('error');
    },
  });

  if (!model || !documents) {
    return null;
  }

  const handleSaveDocument = (data: any) => {
    if (selectedDocument) {
      updateDocument({ ...data, id: selectedDocument.id });
    } else {
      createDocument(data);
    }
  };

  const handleDeleteDocument = (documentId: string) => {
    if (confirm('Are you sure you want to delete this document?')) {
      deleteDocument(documentId);
    }
  };

  const markdownDocs = documents.filter((doc) => doc.type === 'markdown');
  const otherDocs = documents.filter((doc) => doc.type !== 'markdown');

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Documentation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage documentation for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Card>
        <CardContent>
          <Stack spacing={3}>
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
            >
              <Tabs
                value={tabValue}
                onChange={(_, newValue) => setTabValue(newValue)}
                aria-label="documentation tabs"
              >
                <Tab label="Documentation" />
                <Tab label="Files" />
              </Tabs>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => {
                  setSelectedDocument(undefined);
                  setDocumentDialogOpen(true);
                }}
              >
                Add Document
              </Button>
            </Stack>

            <TabPanel value={tabValue} index={0}>
              <List>
                {markdownDocs.map((doc) => (
                  <div key={doc.id}>
                    <ListItem>
                      <ListItemIcon>
                        <DocIcon />
                      </ListItemIcon>
                      <ListItemText
                        primary={doc.title}
                        secondary={`Last updated ${format(
                          new Date(doc.updated_at),
                          'yyyy-MM-dd HH:mm'
                        )} by ${doc.created_by.name}`}
                      />
                      <ListItemSecondaryAction>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="Edit">
                            <IconButton
                              edge="end"
                              onClick={() => {
                                setSelectedDocument(doc);
                                setDocumentDialogOpen(true);
                              }}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              edge="end"
                              color="error"
                              onClick={() => handleDeleteDocument(doc.id)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider />
                  </div>
                ))}
              </List>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <List>
                {otherDocs.map((doc) => (
                  <div key={doc.id}>
                    <ListItem>
                      <ListItemIcon>
                        <DocIcon />
                      </ListItemIcon>
                      <ListItemText
                        primary={doc.title}
                        secondary={`Last updated ${format(
                          new Date(doc.updated_at),
                          'yyyy-MM-dd HH:mm'
                        )} by ${doc.created_by.name}`}
                      />
                      <ListItemSecondaryAction>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="Download">
                            <IconButton edge="end">
                              <DownloadIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              edge="end"
                              color="error"
                              onClick={() => handleDeleteDocument(doc.id)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider />
                  </div>
                ))}
              </List>
            </TabPanel>
          </Stack>
        </CardContent>
      </Card>

      {/* Document Dialog */}
      <DocumentDialog
        open={documentDialogOpen}
        onClose={() => {
          setDocumentDialogOpen(false);
          setSelectedDocument(undefined);
        }}
        onSave={handleSaveDocument}
        document={selectedDocument}
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