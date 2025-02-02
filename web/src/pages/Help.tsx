import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stack,
  Paper,
  Divider,
  Link,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayCircle as VideoIcon,
  Description as DocIcon,
  School as TutorialIcon,
  QuestionAnswer as FaqIcon,
  LiveHelp as SupportIcon,
  Search as SearchIcon,
  BookmarkBorder as GuideIcon,
  Code as ApiIcon,
  Settings as ConfigIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';

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
      id={`help-tabpanel-${index}`}
      aria-labelledby={`help-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function QuickStartGuide() {
  return (
    <Stack spacing={3}>
      <Typography variant="h6" gutterBottom>
        Quick Start Guide
      </Typography>
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1" gutterBottom>
            1. Setting Up Your Environment
          </Typography>
          <List>
            <ListItem>
              <ListItemIcon>
                <ConfigIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Configure System Settings"
                secondary="Set up your environment variables and system preferences"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <DownloadIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Install Dependencies"
                secondary="Make sure all required dependencies are installed"
              />
            </ListItem>
          </List>
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1" gutterBottom>
            2. Creating Your First Dataset
          </Typography>
          <List>
            <ListItem>
              <ListItemIcon>
                <GuideIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Upload Data"
                secondary="Upload your image dataset through the web interface"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <GuideIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Preprocess Data"
                secondary="Configure and run preprocessing steps on your dataset"
              />
            </ListItem>
          </List>
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant="subtitle1" gutterBottom>
            3. Training Your First Model
          </Typography>
          <List>
            <ListItem>
              <ListItemIcon>
                <GuideIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Configure Model"
                secondary="Set up your model architecture and training parameters"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <GuideIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="Start Training"
                secondary="Launch and monitor your training job"
              />
            </ListItem>
          </List>
        </Stack>
      </Paper>
    </Stack>
  );
}

function VideoTutorials() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Video Tutorials
        </Typography>
      </Grid>
      {[
        {
          title: 'Getting Started with PYRAI',
          duration: '10:25',
          thumbnail: '/tutorials/getting-started.jpg',
          description:
            'Learn the basics of PYRAI and how to set up your first project.',
        },
        {
          title: 'Dataset Management',
          duration: '8:15',
          thumbnail: '/tutorials/dataset-management.jpg',
          description:
            'Comprehensive guide to managing and preprocessing your datasets.',
        },
        {
          title: 'Training Models',
          duration: '15:30',
          thumbnail: '/tutorials/model-training.jpg',
          description:
            'Step-by-step guide to training and fine-tuning your models.',
        },
        {
          title: 'Advanced Features',
          duration: '12:45',
          thumbnail: '/tutorials/advanced-features.jpg',
          description: 'Deep dive into advanced features and optimizations.',
        },
      ].map((video, index) => (
        <Grid item xs={12} sm={6} key={index}>
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Box
                  sx={{
                    position: 'relative',
                    paddingTop: '56.25%',
                    bgcolor: 'grey.100',
                    borderRadius: 1,
                  }}
                >
                  <IconButton
                    sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                    }}
                    color="primary"
                    size="large"
                  >
                    <VideoIcon fontSize="large" />
                  </IconButton>
                  <Typography
                    variant="caption"
                    sx={{
                      position: 'absolute',
                      bottom: 8,
                      right: 8,
                      bgcolor: 'rgba(0, 0, 0, 0.7)',
                      color: 'white',
                      px: 1,
                      borderRadius: 1,
                    }}
                  >
                    {video.duration}
                  </Typography>
                </Box>
                <Typography variant="subtitle1">{video.title}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {video.description}
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

function FAQ() {
  return (
    <Stack spacing={3}>
      <Typography variant="h6" gutterBottom>
        Frequently Asked Questions
      </Typography>
      {[
        {
          question: 'What are the system requirements?',
          answer:
            'PYRAI requires a modern web browser and Python 3.8 or higher. For optimal performance, we recommend at least 8GB RAM and a multi-core processor.',
        },
        {
          question: 'How do I update my model parameters?',
          answer:
            'You can update model parameters through the Settings page. Navigate to Models > Settings and adjust the parameters as needed.',
        },
        {
          question: 'Can I export my trained models?',
          answer:
            'Yes, you can export trained models in various formats including ONNX and TensorFlow SavedModel format.',
        },
        {
          question: 'How do I handle large datasets?',
          answer:
            'PYRAI supports streaming and batch processing for large datasets. Use the preprocessing pipeline to efficiently handle large data volumes.',
        },
        {
          question: 'What types of models are supported?',
          answer:
            'PYRAI supports a wide range of deep learning models including CNNs, transformers, and custom architectures.',
        },
      ].map((item, index) => (
        <Accordion key={index}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>{item.question}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography color="text.secondary">{item.answer}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Stack>
  );
}

function TechnicalSupport() {
  return (
    <Stack spacing={3}>
      <Typography variant="h6" gutterBottom>
        Technical Support
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Community Support</Typography>
              <Typography variant="body2" color="text.secondary">
                Join our community forum for discussions, tips, and peer support.
              </Typography>
              <Button
                variant="outlined"
                startIcon={<QuestionAnswer />}
                component={Link}
                href="https://community.pyrai.xyz"
                target="_blank"
              >
                Visit Forum
              </Button>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Documentation</Typography>
              <Typography variant="body2" color="text.secondary">
                Access our comprehensive documentation and API references.
              </Typography>
              <Button
                variant="outlined"
                startIcon={<DocIcon />}
                component={Link}
                href="https://docs.pyrai.xyz"
                target="_blank"
              >
                View Docs
              </Button>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">GitHub</Typography>
              <Typography variant="body2" color="text.secondary">
                Report issues, contribute code, or explore our open source projects.
              </Typography>
              <Button
                variant="outlined"
                startIcon={<Code />}
                component={Link}
                href="https://github.com/pyraixyz/PYRAI"
                target="_blank"
              >
                View Repository
              </Button>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="subtitle1">Email Support</Typography>
              <Typography variant="body2" color="text.secondary">
                Contact our technical support team for direct assistance.
              </Typography>
              <Button
                variant="outlined"
                startIcon={<SupportIcon />}
                component={Link}
                href="mailto:support@pyrai.xyz"
              >
                Contact Support
              </Button>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Stack>
  );
}

export default function Help() {
  const [tabValue, setTabValue] = useState(0);

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Help Center
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Find guides, tutorials, and answers to common questions about PYRAI.
        </Typography>
      </Box>

      {/* Search Bar */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              bgcolor: 'grey.50',
              borderRadius: 1,
              p: 2,
            }}
          >
            <SearchIcon color="action" />
            <Typography color="text.secondary">
              Search the help center...
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Content */}
      <Card>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            aria-label="help center tabs"
          >
            <Tab
              icon={<GuideIcon />}
              label="Quick Start"
              id="help-tab-0"
              aria-controls="help-tabpanel-0"
            />
            <Tab
              icon={<VideoIcon />}
              label="Video Tutorials"
              id="help-tab-1"
              aria-controls="help-tabpanel-1"
            />
            <Tab
              icon={<FaqIcon />}
              label="FAQ"
              id="help-tab-2"
              aria-controls="help-tabpanel-2"
            />
            <Tab
              icon={<SupportIcon />}
              label="Support"
              id="help-tab-3"
              aria-controls="help-tabpanel-3"
            />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <QuickStartGuide />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <VideoTutorials />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <FAQ />
          </TabPanel>
          <TabPanel value={tabValue} index={3}>
            <TechnicalSupport />
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
} 