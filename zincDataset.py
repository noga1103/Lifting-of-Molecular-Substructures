class ZincDataset(Dataset):
    def __init__(self, data, device) -> None:
        self.data = data
        self.device = device
        self.complexes = [item.complex for item in data]
        print(f"Processing {len(self.complexes)} complexes...")
        start_time = time.time()
        self.x_0, self.x_1, self.x_2 = self._get_features()
        end_time = time.time()
        print(f"Feature extraction took {end_time - start_time:.2f} seconds")
        self.y = self._get_labels()
        print("Getting neighborhood matrices...")
        start_time = time.time()
        self.a0, self.a1, self.coa2, self.b1, self.b2 = self._get_neighborhood_matrix()
        end_time = time.time()
        print(f"Neighborhood matrix computation took {end_time - start_time:.2f} seconds")

    def _get_features(self):
        x_0, x_1, x_2 = [], [], []
        for i, complex in enumerate(self.complexes):
            if i % 100 == 0:
                print(f"Processing complex {i}...")

            nodes = [cell for cell in complex.cells if len(cell) == 1]
            edges = [cell for cell in complex.cells if len(cell) == 2]
            faces = [cell for cell in complex.cells if len(cell) > 2]

            node_features = []
            for node in nodes:
                node_element = next(iter(node))
                degree = sum(1 for edge in edges if node_element in edge)
                node_features.append([float(degree)])
            x_0.append(torch.tensor(node_features, dtype=torch.float, device=self.device))

            edge_features = []
            for edge in edges:
                num_faces = sum(1 for face in faces if edge.issubset(face))
                edge_features.append([float(num_faces)])
            x_1.append(torch.tensor(edge_features, dtype=torch.float, device=self.device))

            face_features = []
            for face in faces:
                num_edges = len(face)
                face_features.append([float(num_edges)])
            x_2.append(torch.tensor(face_features, dtype=torch.float, device=self.device))

        return x_0, x_1, x_2

    def _get_labels(self):
        return torch.tensor([item.log_p for item in self.data], dtype=torch.float, device=self.device)

    def _get_neighborhood_matrix(self):
        a0, a1, coa2, b1, b2 = [], [], [], [], []
        for i, cc in enumerate(self.complexes):
            if i % 100 == 0:
                print(f"Computing neighborhood matrices for complex {i}...")

            a0.append(self._scipy_sparse_to_torch_sparse(cc.adjacency_matrix(0, 1)))
            a1.append(self._scipy_sparse_to_torch_sparse(cc.adjacency_matrix(1, 2)))

            B = cc.incidence_matrix(rank=1, to_rank=2)
            A = B.T @ B
            A.setdiag(0)
            coa2.append(self._scipy_sparse_to_torch_sparse(A))

            b1.append(self._scipy_sparse_to_torch_sparse(cc.incidence_matrix(0, 1)))
            b2.append(self._scipy_sparse_to_torch_sparse(cc.incidence_matrix(1, 2)))

        return a0, a1, coa2, b1, b2

    def _scipy_sparse_to_torch_sparse(self, scipy_sparse_matrix):
        coo = scipy_sparse_matrix.tocoo()
        values = torch.FloatTensor(coo.data)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def num_classes(self):
        return 1  # Regression task, so we return 1

    def channels_dim(self):
        return [self.x_0[0].shape[1], self.x_1[0].shape[1], self.x_2[0].shape[1]]

    def len(self):
        return len(self.complexes)

    def getitem(self, idx):
        return (
            self.x_0[idx],
            self.x_1[idx],
            self.x_2[idx],
            self.a0[idx],
            self.a1[idx],
            self.coa2[idx],
            self.b1[idx],
            self.b2[idx],
            self.y[idx],
        )
