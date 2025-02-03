import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Stack,
  Button,
  IconButton,
  Tooltip,
  TextField,
  Alert,
  Snackbar,
  LinearProgress,
  Tabs,
  Tab,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  Chip,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Description as DocIcon,
  Code as CodeIcon,
  Api as ApiIcon,
  Link as LinkIcon,
  CloudUpload as UploadIcon,
  Download as DownloadIcon,
  Preview as PreviewIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { modelApi } from '@/api/client';

interface Document {
  id: string;
  title: string;
  content: string;
  type: 'guide' | 'api' | 'example';
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
  onSave: (data: Partial<Document>) => void;
  document?: Document;
}

interface DocumentHistoryDialogProps {
  open: boolean;
  onClose: () => void;
  document: Document;
}

const DOCUMENT_TYPES = [
  { label: 'User Guide', value: 'guide', icon: <DocIcon /> },
  { label: 'API Reference', value: 'api', icon: <ApiIcon /> },
  { label: 'Code Example', value: 'example', icon: <CodeIcon /> },
];

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
      type: 'guide',
    }
  );
  const [preview, setPreview] = useState(false);

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        {document ? 'Edit Document' : 'Create Document'}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <Stack direction="row" spacing={2}>
            <TextField
              label="Title"
              value={formData.title}
              onChange={(e) =>
                setFormData({ ...formData, title: e.target.value })
              }
              fullWidth
              required
            />
            <TextField
              select
              label="Type"
              value={formData.type}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  type: e.target.value as Document['type'],
                })
              }
              sx={{ width: 200 }}
              SelectProps={{
                renderValue: (value) => (
                  <Stack direction="row" spacing={1} alignItems="center">
                    {DOCUMENT_TYPES.find((t) => t.value === value)?.icon}
                    <Typography>
                      {DOCUMENT_TYPES.find((t) => t.value === value)?.label}
                    </Typography>
                  </Stack>
                ),
              }}
            >
              {DOCUMENT_TYPES.map((type) => (
                <MenuItem key={type.value} value={type.value}>
                  <ListItemIcon>{type.icon}</ListItemIcon>
                  <ListItemText>{type.label}</ListItemText>
                </MenuItem>
              ))}
            </TextField>
          </Stack>

          <Stack spacing={2}>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
              justifyContent="space-between"
            >
              <Typography variant="subtitle2" color="text.secondary">
                Content
              </Typography>
              <Button
                size="small"
                startIcon={preview ? <EditIcon /> : <PreviewIcon />}
                onClick={() => setPreview(!preview)}
              >
                {preview ? 'Edit' : 'Preview'}
              </Button>
            </Stack>

            {preview ? (
              <Paper
                variant="outlined"
                sx={{ p: 2, minHeight: 400, overflow: 'auto' }}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={materialLight}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {formData.content}
                </ReactMarkdown>
              </Paper>
            ) : (
              <TextField
                multiline
                rows={15}
                value={formData.content}
                onChange={(e) =>
                  setFormData({ ...formData, content: e.target.value })
                }
                fullWidth
                placeholder="Write your document in Markdown format..."
              />
            )}
          </Stack>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={!formData.title || !formData.content}
        >
          {document ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function DocumentHistoryDialog({
  open,
  onClose,
  document,
}: DocumentHistoryDialogProps) {
  const { modelId } = useParams<{ modelId: string }>();
  const { data: history, isLoading } = useQuery({
    queryKey: ['model', modelId, 'document', document.id, 'history'],
    queryFn: () => modelApi.getDocumentHistory(modelId!, document.id),
    enabled: !!modelId && !!document.id,
  });

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Document History</DialogTitle>
      <DialogContent>
        {isLoading ? (
          <LinearProgress />
        ) : (
          <List>
            {history?.map((version, index) => (
              <React.Fragment key={version.id}>
                <ListItem>
                  <ListItemIcon>
                    <HistoryIcon color="action" />
                  </ListItemIcon>
                  <ListItemText
                    primary={format(new Date(version.created_at), 'PPpp')}
                    secondary={`Updated by ${version.created_by.name}`}
                  />
                  <ListItemSecondaryAction>
                    <Button
                      size="small"
                      startIcon={<PreviewIcon />}
                      onClick={() => {
                        // TODO: Show version diff
                      }}
                    >
                      View Changes
                    </Button>
                  </ListItemSecondaryAction>
                </ListItem>
                {index < history.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelDocs() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [selectedType, setSelectedType] = useState<Document['type']>('guide');
  const [documentDialogOpen, setDocumentDialogOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document>();
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch documents
  const { data: documents, isLoading } = useQuery({
    queryKey: ['model', modelId, 'documents', selectedType],
    queryFn: () =>
      modelApi.getModelDocuments(modelId!, {
        type: selectedType,
      }),
    enabled: !!modelId,
  });

  // Create document mutation
  const { mutate: createDocument } = useMutation({
    mutationFn: (data: Partial<Document>) =>
      modelApi.createModelDocument(modelId!, data),
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
    mutationFn: (data: Partial<Document>) =>
      modelApi.updateModelDocument(modelId!, data.id!, data),
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
      modelApi.deleteModelDocument(modelId!, documentId),
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

  if (!model || (!documents && !isLoading)) {
    return null;
  }

  const handleDocumentSave = (data: Partial<Document>) => {
    if (selectedDocument) {
      updateDocument({ ...data, id: selectedDocument.id });
    } else {
      createDocument(data);
    }
    setDocumentDialogOpen(false);
    setSelectedDocument(undefined);
  };

  const handleDocumentDelete = (documentId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this document? This action cannot be undone.'
      )
    ) {
      deleteDocument(documentId);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Documentation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage documentation and API references for {model.name}
        </Typography>
      </Box>

      {/* Content */}
      <Stack spacing={3}>
        {/* Document Type Tabs */}
        <Paper variant="outlined">
          <Tabs
            value={selectedType}
            onChange={(e, value) => setSelectedType(value)}
            variant="fullWidth"
          >
            {DOCUMENT_TYPES.map((type) => (
              <Tab
                key={type.value}
                value={type.value}
                label={
                  <Stack direction="row" spacing={1} alignItems="center">
                    {type.icon}
                    <Typography>{type.label}</Typography>
                  </Stack>
                }
              />
            ))}
          </Tabs>
        </Paper>

        {/* Document List */}
        <Stack spacing={2}>
          <Stack
            direction="row"
            spacing={2}
            alignItems="center"
            justifyContent="space-between"
          >
            <Typography variant="h6">
              {DOCUMENT_TYPES.find((t) => t.value === selectedType)?.label}s
            </Typography>
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

          {isLoading ? (
            <LinearProgress />
          ) : (
            <Stack spacing={2}>
              {documents?.map((doc) => (
                <Card key={doc.id} variant="outlined">
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack
                        direction="row"
                        spacing={2}
                        alignItems="center"
                        justifyContent="space-between"
                      >
                        <Stack direction="row" spacing={1} alignItems="center">
                          {DOCUMENT_TYPES.find((t) => t.value === doc.type)?.icon}
                          <Typography variant="h6">{doc.title}</Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Tooltip title="View History">
                            <IconButton
                              onClick={() => {
                                setSelectedDocument(doc);
                                setHistoryDialogOpen(true);
                              }}
                            >
                              <HistoryIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit">
                            <IconButton
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
                              color="error"
                              onClick={() => handleDocumentDelete(doc.id)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </Stack>

                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            code({ node, inline, className, children, ...props }) {
                              const match = /language-(\w+)/.exec(className || '');
                              return !inline && match ? (
                                <SyntaxHighlighter
                                  style={materialLight}
                                  language={match[1]}
                                  PreTag="div"
                                  {...props}
                                >
                                  {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                              ) : (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              );
                            },
                          }}
                        >
                          {doc.content}
                        </ReactMarkdown>
                      </Paper>

                      <Stack
                        direction="row"
                        spacing={2}
                        alignItems="center"
                        justifyContent="space-between"
                      >
                        <Typography variant="caption" color="text.secondary">
                          Last updated{' '}
                          {format(new Date(doc.updated_at), 'PPp')} by{' '}
                          {doc.created_by.name}
                        </Typography>
                        <Stack direction="row" spacing={1}>
                          <Button
                            size="small"
                            startIcon={<DownloadIcon />}
                            onClick={() => {
                              // TODO: Download document
                            }}
                          >
                            Download
                          </Button>
                          <Button
                            size="small"
                            startIcon={<LinkIcon />}
                            onClick={() => {
                              // TODO: Copy document link
                            }}
                          >
                            Share
                          </Button>
                        </Stack>
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Stack>
          )}
        </Stack>
      </Stack>

      {/* Document Dialog */}
      <DocumentDialog
        open={documentDialogOpen}
        onClose={() => {
          setDocumentDialogOpen(false);
          setSelectedDocument(undefined);
        }}
        onSave={handleDocumentSave}
        document={selectedDocument}
      />

      {/* History Dialog */}
      {selectedDocument && (
        <DocumentHistoryDialog
          open={historyDialogOpen}
          onClose={() => {
            setHistoryDialogOpen(false);
            setSelectedDocument(undefined);
          }}
          document={selectedDocument}
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