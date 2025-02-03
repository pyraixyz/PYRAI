import { useState } from 'react';
import { useParams } from 'react-router-dom';
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
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
} from '@mui/material';
import {
  ContentCopy as CopyIcon,
  Key as KeyIcon,
  Code as CodeIcon,
  Description as DocsIcon,
  Settings as ConfigIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
      id={`api-tabpanel-${index}`}
      aria-labelledby={`api-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface EndpointInfoProps {
  modelId: string;
  endpoint: string;
  method: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    required: boolean;
    description: string;
  }>;
}

function EndpointInfo({
  modelId,
  endpoint,
  method,
  description,
  parameters,
}: EndpointInfoProps) {
  return (
    <Stack spacing={3}>
      <Box>
        <Stack direction="row" spacing={2} alignItems="center">
          <Chip
            label={method}
            color={method === 'POST' ? 'primary' : 'default'}
            size="small"
          />
          <Typography variant="subtitle1" component="code">
            {endpoint}
          </Typography>
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {description}
        </Typography>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Parameter</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Required</TableCell>
              <TableCell>Description</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {parameters.map((param) => (
              <TableRow key={param.name}>
                <TableCell component="th" scope="row">
                  {param.name}
                </TableCell>
                <TableCell>
                  <Chip label={param.type} size="small" variant="outlined" />
                </TableCell>
                <TableCell>
                  {param.required ? (
                    <Chip label="Required" size="small" color="error" />
                  ) : (
                    <Chip label="Optional" size="small" />
                  )}
                </TableCell>
                <TableCell>{param.description}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Stack>
  );
}

interface CodeExampleProps {
  language: string;
  code: string;
}

function CodeExample({ language, code }: CodeExampleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}>
        <Tooltip title={copied ? 'Copied!' : 'Copy code'}>
          <IconButton size="small" onClick={handleCopy}>
            <CopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <SyntaxHighlighter
        language={language}
        style={materialDark}
        customStyle={{
          margin: 0,
          borderRadius: 8,
          padding: '16px 48px 16px 16px',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </Box>
  );
}

interface APIKeyDialogProps {
  open: boolean;
  onClose: () => void;
  onGenerate: () => void;
}

function APIKeyDialog({ open, onClose, onGenerate }: APIKeyDialogProps) {
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Generate New API Key</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          This will generate a new API key for accessing the model endpoints. The
          old key will be invalidated immediately.
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={onGenerate} variant="contained" color="primary">
          Generate
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelAPI() {
  const { modelId } = useParams<{ modelId: string }>();
  const [tabValue, setTabValue] = useState(0);
  const [keyDialogOpen, setKeyDialogOpen] = useState(false);

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch API key
  const { data: apiKey } = useQuery({
    queryKey: ['model', modelId, 'apiKey'],
    queryFn: () => modelApi.getAPIKey(modelId!),
    enabled: !!modelId,
  });

  // Generate API key mutation
  const { mutate: generateKey } = useMutation({
    mutationFn: () => modelApi.generateAPIKey(modelId!),
    onSuccess: () => {
      setKeyDialogOpen(false);
    },
  });

  if (!model) {
    return null;
  }

  const endpoints = [
    {
      method: 'POST',
      endpoint: `/api/v1/models/${modelId}/predict`,
      description: 'Make a single prediction with the model',
      parameters: [
        {
          name: 'input',
          type: 'object',
          required: true,
          description: 'Input data for prediction',
        },
        {
          name: 'api_key',
          type: 'string',
          required: true,
          description: 'API key for authentication',
        },
      ],
    },
    {
      method: 'POST',
      endpoint: `/api/v1/models/${modelId}/predict_batch`,
      description: 'Make batch predictions with the model',
      parameters: [
        {
          name: 'inputs',
          type: 'array',
          required: true,
          description: 'Array of input data for predictions',
        },
        {
          name: 'api_key',
          type: 'string',
          required: true,
          description: 'API key for authentication',
        },
      ],
    },
  ];

  const examples = {
    python: `import requests

API_KEY = "${apiKey || 'your_api_key'}"
MODEL_ID = "${modelId}"

# Single prediction
response = requests.post(
    f"https://api.example.com/api/v1/models/{MODEL_ID}/predict",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"input": {"feature1": 1.0, "feature2": "value"}}
)
result = response.json()
print(result)

# Batch prediction
response = requests.post(
    f"https://api.example.com/api/v1/models/{MODEL_ID}/predict_batch",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"inputs": [
        {"feature1": 1.0, "feature2": "value1"},
        {"feature1": 2.0, "feature2": "value2"}
    ]}
)
results = response.json()
print(results)`,
    javascript: `const API_KEY = "${apiKey || 'your_api_key'}";
const MODEL_ID = "${modelId}";

// Single prediction
const response = await fetch(
  \`https://api.example.com/api/v1/models/\${MODEL_ID}/predict\`,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': \`Bearer \${API_KEY}\`
    },
    body: JSON.stringify({
      input: { feature1: 1.0, feature2: 'value' }
    })
  }
);
const result = await response.json();
console.log(result);

// Batch prediction
const batchResponse = await fetch(
  \`https://api.example.com/api/v1/models/\${MODEL_ID}/predict_batch\`,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': \`Bearer \${API_KEY}\`
    },
    body: JSON.stringify({
      inputs: [
        { feature1: 1.0, feature2: 'value1' },
        { feature1: 2.0, feature2: 'value2' }
      ]
    })
  }
);
const results = await batchResponse.json();
console.log(results);`,
    curl: `# Single prediction
curl -X POST \\
  "https://api.example.com/api/v1/models/${modelId}/predict" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey || 'your_api_key'}" \\
  -d '{
    "input": {
      "feature1": 1.0,
      "feature2": "value"
    }
  }'

# Batch prediction
curl -X POST \\
  "https://api.example.com/api/v1/models/${modelId}/predict_batch" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey || 'your_api_key'}" \\
  -d '{
    "inputs": [
      {
        "feature1": 1.0,
        "feature2": "value1"
      },
      {
        "feature1": 2.0,
        "feature2": "value2"
      }
    ]
  }'`,
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          API Documentation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Integration guide and examples for {model.name}
        </Typography>
      </Box>

      {/* API Key Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack
            direction="row"
            spacing={2}
            alignItems="center"
            justifyContent="space-between"
          >
            <Stack direction="row" spacing={2} alignItems="center">
              <KeyIcon color="primary" />
              <div>
                <Typography variant="subtitle1">API Key</Typography>
                <Typography variant="body2" color="text.secondary">
                  {apiKey
                    ? `${apiKey.substring(0, 8)}...${apiKey.substring(
                        apiKey.length - 8
                      )}`
                    : 'No API key generated'}
                </Typography>
              </div>
            </Stack>
            <Button
              variant="contained"
              onClick={() => setKeyDialogOpen(true)}
              startIcon={<KeyIcon />}
            >
              Generate New Key
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            aria-label="api documentation tabs"
          >
            <Tab
              icon={<DocsIcon />}
              label="Documentation"
              id="api-tab-0"
              aria-controls="api-tabpanel-0"
            />
            <Tab
              icon={<CodeIcon />}
              label="Code Examples"
              id="api-tab-1"
              aria-controls="api-tabpanel-1"
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Stack spacing={4}>
              {endpoints.map((endpoint, index) => (
                <Box key={index}>
                  {index > 0 && <Divider sx={{ my: 4 }} />}
                  <EndpointInfo
                    modelId={modelId!}
                    endpoint={endpoint.endpoint}
                    method={endpoint.method}
                    description={endpoint.description}
                    parameters={endpoint.parameters}
                  />
                </Box>
              ))}
            </Stack>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Stack spacing={4}>
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Python
                </Typography>
                <CodeExample language="python" code={examples.python} />
              </Box>
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  JavaScript
                </Typography>
                <CodeExample language="javascript" code={examples.javascript} />
              </Box>
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  cURL
                </Typography>
                <CodeExample language="bash" code={examples.curl} />
              </Box>
            </Stack>
          </TabPanel>
        </CardContent>
      </Card>

      {/* API Key Dialog */}
      <APIKeyDialog
        open={keyDialogOpen}
        onClose={() => setKeyDialogOpen(false)}
        onGenerate={() => generateKey()}
      />
    </Box>
  );
} 