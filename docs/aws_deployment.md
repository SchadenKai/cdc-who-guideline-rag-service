## AWS Deployment Guide
Includes step by step guide on how to setup, startup, and takedown your AWS setup needed in order to deploy this project securely, reliably, and scalable.

## Initial Setup
Steps that will only be followed once to setup and prepare your AWS account to host the project. Once done, you don't need to perform the following steps again.
> Note: This setup will incur a recurring cost, though minimal, due to storage being used in AWS. All of the steps that will be done will include creating and storing resources in AWS that will be needed later during startup. Refer to the [cost breakdown](./cloud_architecture.md) specifically in the monthly recurring cost on the estimated cost for this project.

### Resources List
- Cloudflare DNS
- AWS Target Group
- AWS Security Group
- AWS VPC
- AWS Task Definition
- AWS SSM Parameter Store
  
### Steps
1. **Cloudflare DNS Setup**: Create an account in Cloudflare and register your DNS name
2. **Security Groups Setup:** Create two security groups. These security groups will be used for the load balancer and for the task containers where the project will be running
   1. Load Balancer (ALB) Security Group: Set the inbound rule to allow all IPs from HTTP protocol for port 80. Then outbound rule to allow HTTP protocol for port 80 to be sent to any IPv4 (You can change this later to only point to the application's security group once done)
   2. Application Security Group (for tasks / services): Set the inbound role to accept HTTP protocol from port 80 from a custom IP of the load balancer's security group. Then set the outbound to any for all settings (protocol, port, and ips) to support connecting to the internet and connect to third party applications and to enable web scrapping.
   3. (optional) Further security: Update the load balancer security group's outbound rule HTTP protocol to only send to custom IP of the application security group. This is so that we have refined control over what services the load balancer can only route to.
3. **VPC Setup:** Create a virtual private cloud and create 4 subnets, 2 for public and 2 for private. The public subnets are where the load balancer and NAT gateway will reside while the private subnets are where you application tasks and services will reside.
   1. When creating, click the "VPC and more" in order to specify the number of availability zones, and the number private and public subnets.
   2. Specify the number of Availability Zones (AZs). It is recommended to have more than 1 since we will be working on load balancer which requires it.
   3. Specify the number of public and private subnets. Since we just have the Load Balancer and NAT Gateway as our gateways to the internet, we just need to have 2 pubic subnets to hosts those services which must have an access to the public network. Then we create two private subnets, 1 for each AZ in order to host our ECS tasks.
   4. During creation, enable the DNS hostname and DNS resolution. This is what the DNS provider will use (which is Cloudflare) in order to communicate with the load balancer. In itself, the DNS name already is a functional link to unsecurely access your application after startup through HTTP.
4. **AWS Target Group:** This will be used in order for your services to be reachable by the load balancer. Target gropus will be the one responsible for specifying the list of destination a request can go to. Note that targets in a target group are treated equally by the load balancer, thus, if you have a web app target in a target group for api server app, the load balancer will treat it as if it is an api server. Since we are just working on the api server right now, we just need to create one target group for the api server. Once we work on the web server, we then need to specify another target group for the web server.
   1. Since we are working with ECS Fargate, we will be using target type of IP addresses to make use of IPs, in this case private IPs, to route traffic into. 
   2. We then need to specify the protocol and port on where and how the load balancer will route the traffic into. We will set it to HTTP and port 8000 since this is where our application resides and what our application uses. If our app makes use of a different protocol (ex. TCP) and in a different port (ex. 3000) which must be specified in the task definition, then we must configure the target group accordingly.
5. **AWS SSM Parameter Store:**: Used to store our secret environment variables which contains sensitive information such as API keys and other credentials.
   1. Go to SSM > under the Application tools, click on Parameter Store
   2. Click on "Create Parameter" in order to start creating one.
   3. Make sure to change the type of the parameter from "String" (default) into "SecureString" in order for the data to be encrypted. If you have a parameter value that is a list of strings separated by comma, you can then make use of type "StringList".
   4. Enter the name and the value in their respective fields, then create.
6. **AWS Task Definition:** Used to specify how our application will be deployed inside a machine and we can expose that application inside that machine. 
   1. Specify the launch type: This depends on the use case of the project. AWS Fargate is a serverless way of deploying our application, thus, it is priced based on the amount of time it is running. AWS ECS instance is priced per hour the application is running. Since we don't have long running tasks or 24/7 running cron jobs, we also don't need to make use of GPU's or need to host our own LLM, and is only serving response per requests, then Fargate is the best for cost. 
   2. Specifying the OS and architecture: We can keep the OS in default, making use of Linux with x86_64 architecture. Then vCPU of 1 and 3gb of memory, which is already sufficient enough to deploy our app. Since we are also making use of Fargate, we don't need to think about allocating extra resources for extra sidecars running for AWS such as ECS Agent, Docker, etc.
   3. Specify the task role: This will be the role of the ECS agent working on creating and managing your task. Default role can do but since we are making use of a parameter store which we will use to store and retrieve environment variables, we need to add extra permissions to the default role.
      1. Go to IAM
      2. Go to Roles and add a permission of the default role for ECS agent called `ecsTaskExecutionRole`
      3. Add the permission called `AmazonSSMReadOnly` which would enable the role to read the parameters (environment variables) stored in AWS SSM.
      4. Create a container. In default, there will be a Container 1 already created which is the essential / required container. You just need to specify the docker image of the app, in this case `schadenkai/cdc-who-rag-system:latest` which is hosted in Docker Hub. If it is hosted in ECR, you can just select in in the "Browse ECR Images".
      5. Create a port mapping where you will just need to specify the application port and protocol being used by the app. This would enable us to expose the port of the application outside of the machine to be acessible through the network (which is this case is `awsvpc`). Since the app is running in port 8000 based on the spefied port in the Dockerfile, we will just it to port 8000 with HTTP protocol. 
      6. Then we just need to specify the resource allocated for that specific service. The application is non resource intensive, thus can even run in 0.5 vCPU and 1GB memory.
      7. We then need to specify the environment variables to be used by our app. We can set non sensitive variables in the task definition itself by setting it the value type to "Value". For our sensitive secret environment variables that are residing in Parameter Store, we need to get the ARN of the specific variable and paste it as a value then also changing the value type for the key into "Value From".
      8. Modify the default health check. Intead of calling the "/" route, we need to change it into the "/health" route which is available in our api server. You can also adjust the interval, timeout, and start period as needed in order to make the health check successful.
## Startup
This includes the step by step process to be done every time you need to deploy the application in AWS.
1. Create and attach a NAT Gateway to the VPC that you have. 
   1. Set the availability zone to Zonal
   2. Select a subnet on where the NAT gateway will be created and resides
   3. Allocate an Elastic IP needed for a NAT gateway 
2. Create an application load balancer with the VPC that you created specifically in the public subnets. Then make use of the target group thta you have as the target group for your load balancer. Also, make use of the security group that you created for your load balancer that only accepts traffic from Cloudflare to port 80 (HTTP). 
3. Create a cluster where you will be deploying your application. Clusters are just group of tasks and are ideally grouped by environment. When creating a cluster, you just need to specify what infrastructure will be deployed in it (ex. Fargate only, Fargate and Managed Instances, Fargate and Self-managed Instances). Since we will just be using Fargate, you can just choose to create Fargate only.
4. Create an ECS Service for the task definition created at the start inside the cluster that you created. This will be the setup for the deployment of the app. 
   1. You will need to set the computation configuration, which determines the strategy on how you deploy and allocate compute resources for your tasks. You can just set this to "Launch Type"
   2. (optional) You can also specify the number of replicas you wanted to spin up, specifying the number of tasks duplicates there will be running. This is useful if you expect a lot of traffic and wanted to increase the availability and consistency of the app.
   3. Specify the networking to make use of the VPC you created at the initial setup. Make sure to only use subnets that are private so that tasks will only be deployed in those private subnets.
   4. Make sure to also turn off the public IP
   5. Select the security group that you created for your application. This security group must only receive traffic from the load balancer's security group
   6. Under load balancing, specify the load balancer and target group you created.
   7. (optional) You can also specify the deployment strategy in case where you wanted to rollout new versions of the app. In default, it is set to rolling update where old tasks will be replaced with new ones one at a time. There are also other options such as Blue/Green deployment, Canary, and Linear.
## Takedown
Delete the following resources every time you are done making use of the app in order to limit and reduce the total cost.
1. Delete the AWS ECS Service that is running your application in the cluster
2. Delete the application load balancer
3. Delete the NAT gateway 
4. Delete the elastic IPs associated with your NAT gateway