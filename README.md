# FedE &mdash; Federated learning Emulator
The *Federated learning Emulator* (**FedE**) is a microservice-based SoftWare (SW) toolkit for the middleware-oriented customizable emulation of ground-aerial semi-asynchronous/asynchronous FL applications.

## About
**FedE** is a toolkit that implements all the functionalities needed for the emulation of ground-aerial semi-asynchronous/asynchronous FL applications. The emulated functionalities include: 
1. mobility models of the participating *Coworkers* (CoWs); 
2. communication services at the Physical, MAC and middleware layers;
3. Machine Learning (ML) model compression at the middleware layer;
4. customization of the aggregation and local updating schemes featuring the actually implemented Federated Learning (FL) application; and, 
5. per-CoW partition of the training data, just to name a few.

All these functionalities are *standing-alone microservices*, which communicate through suitably-designed APIs. Since microservices must be, by definition, independently deployable, this allows **FedE** users to build up customized FL emulation environments through the composition of the provided microservices.

**FedE** limits to provide methods to support SPMD parallel programming of distributed-memory process-based codes, whereas it utilizes MPI-based blocking synchronous constructs (like as, for example, `spmdSend`, `spmdReceive`, `spmdBroadcast`, `spmdSendReceive`, `spmdBarrier`) for implementing inter-process communication. To guarantee reliable emulations of real-world multi-node distributed FL applications, we implemented the orchestrator and each coworker as a software process, which is hosted by a dedicated computing core and is equipped with dedicated resources. SPMD programming is used to support inter-process parallelism. 


## MATLAB requirements
It is necessary to install the **Parallel Computing Toolbox** of MATLAB. At least version **R2022b** is necessary.

To run **FedE**, you can run the main file:
```
FedE.m
```


## License
The code in this repository, including all code samples in the notebooks listed above, is released under the [Apache License, Version 2.0](LICENSE). Read more at the [Open Source Initiative](https://opensource.org/license/apache-2.0).
