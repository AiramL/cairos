import torch

class Distiller(torch.nn.Module):

    def __init__(self, 
                 student, 
                 teacher, 
                 alpha=0.1, 
                 temperature=3.0,
                 logger=None):
        
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        self.logger = logger

    def forward(self, 
                x):

        return self.student(x)

    def compute_loss(self, 
                     x, 
                     y):

        with torch.no_grad():

            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        # Distillation loss
        student_soft = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        loss_kd = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Supervised loss
        loss_ce = self.ce_loss(student_logits, y)

        # Combined loss
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        return loss, student_logits

    def fit(self, 
            train_loader, 
            optimizer, 
            epochs=10, 
            device="cuda"):

        self.student.to(device)
        self.teacher.to(device)
        self.teacher.eval()
        self.student.train()

        for epoch in range(1, epochs + 1):

            total_loss = 0.0
            correct = 0
            total = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, logits = self.compute_loss(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

            acc = correct / total
            avg_loss = total_loss / total
            self.logger.debug(f"[SingleTeacher] Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

class MultiTeacherDistiller(Distiller):

    def __init__(self, 
                 student, 
                 teachers, 
                 alpha=0.1, 
                 temperature=3.0,
                 logger=None):
        
        super().__init__(student, 
                         None, 
                         alpha, 
                         temperature,
                         logger)  
        
        self.teachers = teachers

    def compute_loss(self, 
                     x, 
                     y):

        teacher_logits_list = []

        with torch.no_grad():
        
            for teacher in self.teachers:
        
                teacher_logits_list.append(teacher(x))

        avg_teacher_logits = torch.mean(torch.stack(teacher_logits_list), dim=0)

        student_logits = self.student(x)

        # Distillation loss
        student_soft = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.nn.functional.softmax(avg_teacher_logits / self.temperature, dim=1)
        loss_kd = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Supervised loss
        loss_ce = self.ce_loss(student_logits, y)

        # Combined loss
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        return loss, student_logits

    def fit(self, 
            train_loader, 
            optimizer, 
            epochs=10, 
            device="cuda"):

        self.student.to(device)
        for teacher in self.teachers:
            teacher.to(device)
            teacher.eval()
        self.student.train()

        for epoch in range(1, epochs + 1):

            total_loss = 0.0
            correct = 0
            total = 0

            for x, y in train_loader:
                
                x, y = x.to(device), y.to(device)
                loss, logits = self.compute_loss(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

            acc = correct / total
            avg_loss = total_loss / total
            self.logger.debug(f"[MultiTeacher] Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
