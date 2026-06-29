import numpy as np

class PETScGatherHEOMRHS:
    """ A class for collecting elements of the right-hand side matrix
        of the HEOM and streaming them directly into a distributed PETSc Matrix
        to avoid Python object memory overhead.
    """
    def __init__(self, f_idx, block, nhe):
        self._block_size = block
        self._n_blocks = nhe
        self._f_idx = f_idx
        
        try:
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("petsc4py is required for the PETSc backend.")
            
        comm = PETSc.COMM_WORLD
        size = comm.getSize() # Number of ranks
        rank = comm.getRank() # Rank id  and if rank == 0, it is the maser node
        
        #The math of splitting the work 
        #Total ADOs (nhe = 1,000)
        global_size = block * nhe
        n_local_blocks = nhe // size   # 1,000 // 4 = 250
        remainder = nhe % size         # 1,000 % 4 = 0
        
        #  Rank 0 is assigned ADOs 0 to 249.
        #  Rank 1 is assigned ADOs 250 to 499.
        #  Rank 2 is assigned ADOs 500 to 749.
        #  Rank 3 is assigned ADOs 750 to 999.
        if rank < remainder:
            local_blocks = n_local_blocks + 1
        else:
            local_blocks = n_local_blocks
            
        local_size = local_blocks * block
        
        #Creating the distributed matrix
        self.mat = PETSc.Mat().create(comm)
        self.mat.setSizes(((local_size, global_size), (local_size, global_size)))
        self.mat.setType(PETSc.Mat.Type.MPIAIJ) # tells the PETSc to create a distributed matrix
        # This tells PETSc: "Do not create a normal matrix in local RAM." Instead,
        # it creates a Parallel Compressed Sparse Row matrix over the network. Rank 0 only holds rows 0-249 in its local
        # RAM. Rank 1 only holds rows 250-499, and so on.

        # Preallocation estimate
        max_connections = 50 * block
        self.mat.setPreallocationNNZ((max_connections, max_connections))
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    #This maps an ADO coupling term directly into PETSc matrix 
    def add_op(self, row_he, col_he, op):
        from petsc4py import PETSc
        #Convert the ADO label (like (1,0,0)) into an integer ID (like 249)
        row_blk = self._f_idx(row_he)
        col_blk = self._f_idx(col_he)
        
        # Calculate the exact row and column pixel coordinates in the global matrix
        row_indices = np.arange(row_blk * self._block_size, (row_blk + 1) * self._block_size, dtype=np.int32)
        col_indices = np.arange(col_blk * self._block_size, (col_blk + 1) * self._block_size, dtype=np.int32)
        #Inject the data
        self.mat.setValues(row_indices, col_indices, op.as_scipy().todense(), addv=PETSc.InsertMode.ADD_VALUES)

    def gather(self, L_sys=None):
        from petsc4py import PETSc
        if L_sys is not None and L_sys.isconstant:
            L_sys_dense = L_sys(0).data.as_scipy().todense()
            
            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()
            
            n_local_blocks = self._n_blocks // size
            remainder = self._n_blocks % size
            if rank < remainder:
                start_block = rank * (n_local_blocks + 1)
                end_block = start_block + n_local_blocks + 1
            else:
                start_block = rank * n_local_blocks + remainder
                end_block = start_block + n_local_blocks
            
            for r_blk in range(start_block, end_block):
                row_indices = np.arange(r_blk * self._block_size, (r_blk + 1) * self._block_size, dtype=np.int32)
                self.mat.setValues(row_indices, row_indices, L_sys_dense, addv=PETSc.InsertMode.ADD_VALUES)
                
        self.mat.assemblyBegin()
        self.mat.assemblyEnd()
        return self.mat
