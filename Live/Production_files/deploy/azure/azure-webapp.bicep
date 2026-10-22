// Optional reference: the same resources deploy.sh creates imperatively,
// expressed as Infrastructure as Code. Not required for the live session —
// point students here afterwards as the "how do real teams do this" answer.
//
// Usage:
//   az deployment group create \
//     --resource-group rg-multidoc-assist \
//     --template-file azure-webapp.bicep \
//     --parameters acrName=acrmultidocassist webAppName=multidoc-assist-demo \
//                  openAiApiKey=$OPENAI_API_KEY cohereApiKey=$COHERE_API_KEY \
//                  openAiModel=gpt-4o-mini
//
// Note: this provisions the registry + plan + web app, but does NOT build or
// push an image (that still needs `az acr build`, same as deploy.sh step 3) —
// run that first, or the web app will have nothing to pull.

@description('Globally unique name for the Azure Container Registry')
param acrName string

@description('Globally unique name for the Web App (becomes <webAppName>.azurewebsites.net)')
param webAppName string

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('App Service plan SKU')
param skuName string = 'B1'

@description('Image tag to deploy')
param imageTag string = 'v1'

@secure()
param openAiApiKey string

@secure()
param cohereApiKey string

@description('llm.py reads this directly and 400s ("you must provide a model parameter") if unset')
param openAiModel string = 'gpt-4o-mini'

param openAiEmbeddingModel string = ''
param langsmithApiKey string = ''
param langchainProject string = 'multidoc-assist-azure'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: '${webAppName}-plan'
  location: location
  sku: { name: skuName }
  kind: 'linux'
  properties: { reserved: true }
}

resource webApp 'Microsoft.Web/sites@2023-12-01' = {
  name: webAppName
  location: location
  properties: {
    serverFarmId: plan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|${acr.properties.loginServer}/multidoc-assist:${imageTag}'
      appSettings: [
        { name: 'WEBSITES_PORT', value: '8080' }
        { name: 'DOCKER_REGISTRY_SERVER_URL', value: 'https://${acr.properties.loginServer}' }
        { name: 'DOCKER_REGISTRY_SERVER_USERNAME', value: acr.listCredentials().username }
        { name: 'DOCKER_REGISTRY_SERVER_PASSWORD', value: acr.listCredentials().passwords[0].value }
        { name: 'OPENAI_API_KEY', value: openAiApiKey }
        { name: 'COHERE_API_KEY', value: cohereApiKey }
        { name: 'OPENAI_MODEL', value: openAiModel }
        { name: 'OPENAI_EMBEDDING_MODEL', value: openAiEmbeddingModel }
        { name: 'LANGSMITH_API_KEY', value: langsmithApiKey }
        { name: 'LANGCHAIN_PROJECT', value: langchainProject }
      ]
    }
  }
}

output webAppUrl string = 'https://${webApp.properties.defaultHostName}'
output acrLoginServer string = acr.properties.loginServer
